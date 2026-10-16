import assert from "node:assert/strict";
import test from "node:test";
import {
  createSegmenter, normalizeInput, normalizeOutput, erodeMask, refineMaskGPU, SEGMENTATION_APPROXIMATIONS,
} from "../js/pipeline/segment.js";

class Canvas {
  constructor(width, height, fill = 0) {
    this.width = width;
    this.height = height;
    this.data = new Uint8ClampedArray(width * height * 4).fill(fill);
  }
  getContext() {
    const self = this;
    return {
      createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData: (image) => { self.data = image.data.slice(); },
      getImageData: () => ({ data: self.data.slice() }),
      drawImage(source) {
        for (let y = 0; y < self.height; y++) {
          for (let x = 0; x < self.width; x++) {
            const offset = (Math.floor(y / self.height * source.height) * source.width +
              Math.floor(x / self.width * source.width)) * 4;
            self.data.set(source.data.subarray(offset, offset + 4), (y * self.width + x) * 4);
          }
        }
      },
    };
  }
}

function runtime({ run, create } = {}) {
  return {
    Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } },
    InferenceSession: {
      async create(bytes, options) {
        if (create) await create(bytes, options);
        return {
          inputNames: ["input"], outputNames: ["first"],
          async run(feeds) {
            if (run) await run(bytes, feeds, options);
            const data = Float32Array.from({ length: 320 * 320 }, (_, i) => i % 320);
            return { first: { data, dims: [1, 1, 320, 320] } };
          },
          async release() {},
        };
      },
    },
  };
}

test("input uses rembg global maximum and NCHW mean/std, including black input", () => {
  const result = normalizeInput(new Uint8Array([100, 50, 0, 255, 0, 25, 75, 255]));
  const expected = [(1 - 0.485) / 0.229, -0.485 / 0.229,
    (0.5 - 0.456) / 0.224, (0.25 - 0.456) / 0.224,
    -0.406 / 0.225, (0.75 - 0.406) / 0.225];
  expected.forEach((value, i) => assert.ok(Math.abs(result[i] - value) < 1e-6));
  assert.ok(normalizeInput(new Uint8Array(4)).every(Number.isFinite));
});

test("first-output normalization handles constant and invalid predictions", () => {
  assert.deepEqual([...normalizeOutput([2, 4, 6])], [0, 127, 255]);
  assert.deepEqual([...normalizeOutput([3, 3])], [0, 0]);
  assert.throws(() => normalizeOutput([NaN]), /non-finite/);
});

test("linear-time trim equals a clamped square minimum filter", () => {
  const w = 9;
  const h = 7;
  const values = Uint8ClampedArray.from({ length: w * h }, (_, i) => i * 47 % 256);
  for (const radius of [0, 1, 3, 9]) {
    const expected = values.map((_, i) => {
      let min = 255;
      for (let y = Math.max(0, Math.floor(i / w) - radius); y <= Math.min(h - 1, Math.floor(i / w) + radius); y++) {
        for (let x = Math.max(0, i % w - radius); x <= Math.min(w - 1, i % w + radius); x++) min = Math.min(min, values[y * w + x]);
      }
      return min;
    });
    assert.deepEqual(erodeMask(values, w, h, radius), expected);
  }
  assert.match(SEGMENTATION_APPROXIMATIONS.edgeRefinement, /WebGL2 guided filter/);
});

test("small model returns before large download; cached sessions upgrade independently", async () => {
  let resolveLarge;
  const large = new Promise((resolve) => { resolveLarge = resolve; });
  const loads = [];
  const progress = [];
  let upgraded;
  const ready = new Promise((resolve) => { upgraded = resolve; });
  const segmenter = createSegmenter({
    createCanvas: (w, h) => new Canvas(w, h),
    runtimeLoader: async () => runtime({
      run(name, feeds) {
        assert.deepEqual(feeds.input.dims, [1, 3, 320, 320]);
        assert.equal(feeds.input.data.length, 307200);
      },
    }),
    modelLoader: async (name) => {
      loads.push(name);
      if (name === "u2net") await large;
      return name;
    },
  });
  const photo = new Canvas(11, 7, 128);
  const mask = await segmenter.segmentPhoto(photo, {
    trim: 0, onProgress: (event) => progress.push(event), onUpgrade: upgraded,
  });
  assert.equal(mask.width, 11);
  assert.equal(mask.height, 7);
  assert.equal(mask.data[0], 0);
  assert.equal(mask.data[3], 255);
  assert.deepEqual(loads, ["u2netp", "u2net"]);
  resolveLarge();
  const full = await ready;
  assert.equal(full.width, 11);
  await segmenter.warmModels();
  assert.equal(loads.length, 2);
  assert.ok(progress.every((event) => typeof event.stage === "string" &&
    event.progress >= 0 && event.progress <= 1 && typeof event.message === "string"));
});

test("large-model failure reports an actionable upgrade error without rejecting initial result", async () => {
  let report;
  const failed = new Promise((resolve) => { report = resolve; });
  const segmenter = createSegmenter({
    createCanvas: (w, h) => new Canvas(w, h),
    runtimeLoader: async () => runtime(),
    modelLoader: async (name) => {
      if (name === "u2net") throw new Error("Install u2net.onnx");
      return name;
    },
  });
  await segmenter.segmentPhoto(new Canvas(2, 2), {
    trim: 0,
    onProgress: (event) => { if (event.stage === "upgrade-error") report(event); },
    onUpgrade: () => assert.fail("must not fake an upgrade"),
  });
  assert.match((await failed).message, /Install u2net.onnx/);
});

test("WebGPU execution failure retries on WASM", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {} } });
  try {
    const providers = [];
    const segmenter = createSegmenter({
      createCanvas: (w, h) => new Canvas(w, h),
      runtimeLoader: async () => runtime({
        create: (_, options) => providers.push(options.executionProviders[0]),
        run: (_, feeds, options) => {
          if (options.executionProviders[0] === "webgpu") throw new Error("GPU lost");
        },
      }),
      modelLoader: async (name) => name,
    });
    let upgrade;
    const done = new Promise((resolve) => { upgrade = resolve; });
    await segmenter.segmentPhoto(new Canvas(2, 2), { trim: 0, onUpgrade: upgrade });
    await done;
    assert.equal(providers.filter((p) => p === "webgpu").length, 2);
    assert.equal(providers.filter((p) => p === "wasm").length, 2);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "navigator", descriptor);
    else delete globalThis.navigator;
  }
});

test("WebGPU initialization failure falls back; unavailable WASM produces an actionable error", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {} } });
  try {
    const providers = [];
    const segmenter = createSegmenter({
      runtimeLoader: async () => runtime({
        create: (_, options) => {
          const provider = options.executionProviders[0];
          providers.push(provider);
          if (provider === "webgpu") throw new Error("Adapter unsupported");
        },
      }),
      modelLoader: async (name) => name,
    });
    await segmenter.warmModels();
    assert.deepEqual(providers, ["webgpu", "wasm", "webgpu", "wasm"]);
    const broken = createSegmenter({
      runtimeLoader: async () => runtime({ create: () => { throw new Error("Unavailable"); } }),
      modelLoader: async (name) => name,
    });
    await assert.rejects(broken.warmModels(), /allow the pinned Runtime WASM assets/);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "navigator", descriptor);
    else delete globalThis.navigator;
  }
});

test("CacheStorage uses local model URLs and tolerates unavailable storage", async () => {
  const oldFetch = globalThis.fetch;
  const oldCaches = globalThis.caches;
  const stored = new Map();
  let requests = 0;
  globalThis.fetch = async (url) => {
    assert.match(url, /\/frontend\/models\/u2netp?\.onnx$/);
    requests++;
    return new Response(new Uint8Array([1, 2, 3]));
  };
  globalThis.caches = {
    open: async () => ({
      match: async (url) => stored.get(url)?.clone(),
      put: async (url, response) => { stored.set(url, response); },
    }),
  };
  try {
    for (let i = 0; i < 2; i++) {
      const segmenter = createSegmenter({ runtimeLoader: async () => runtime() });
      await segmenter.warmModels();
    }
    assert.equal(requests, 2);
    globalThis.caches.open = async () => { throw new Error("Private mode"); };
    await createSegmenter({ runtimeLoader: async () => runtime() }).warmModels();
    assert.equal(requests, 4);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldCaches === undefined) delete globalThis.caches;
    else globalThis.caches = oldCaches;
  }
});

test("a missing local model names its installation path rather than producing a fake mask", async () => {
  const oldFetch = globalThis.fetch;
  const oldCaches = globalThis.caches;
  globalThis.fetch = async () => new Response("missing", { status: 404 });
  globalThis.caches = undefined;
  try {
    await assert.rejects(createSegmenter({ runtimeLoader: async () => runtime() }).warmModels(),
      /Cannot fetch u2netp\.onnx.*frontend\/models\/u2netp\.onnx/);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldCaches === undefined) delete globalThis.caches;
    else globalThis.caches = oldCaches;
  }
});

test("streams byte download progress with known and unknown response lengths", async () => {
  const oldFetch = globalThis.fetch;
  const oldCaches = globalThis.caches;
  globalThis.caches = undefined;
  let requests = 0;
  globalThis.fetch = async () => {
    const known = requests++ === 0;
    return new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3]));
        controller.enqueue(new Uint8Array([4, 5, 6]));
        controller.close();
      },
    }), { headers: known ? { "content-length": "6" } : {} });
  };
  try {
    const events = [];
    await createSegmenter({ runtimeLoader: async () => runtime() }).warmModels((event) => events.push(event));
    assert.ok(events.some(({ stage, progress }) => stage === "download" && progress === 0.5));
    assert.ok(events.some(({ message }) => /size unknown/.test(message)));
    assert.equal(events.filter(({ stage, progress }) => stage === "download" && progress === 1).length, 2);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldCaches === undefined) delete globalThis.caches;
    else globalThis.caches = oldCaches;
  }
});

test("interrupted streamed download has an actionable retry error", async () => {
  const oldFetch = globalThis.fetch;
  const oldCaches = globalThis.caches;
  globalThis.caches = undefined;
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) { controller.error(new Error("Disconnected")); },
  }));
  try {
    await assert.rejects(createSegmenter({ runtimeLoader: async () => runtime() }).warmModels(),
      /interrupted.*connection and retry/);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldCaches === undefined) delete globalThis.caches;
    else globalThis.caches = oldCaches;
  }
});

test("guided filter reports unsupported float targets and memory budget rather than pretending success", () => {
  const mask = new Canvas(3, 2);
  const unavailable = refineMaskGPU(mask, mask, { createCanvas: () => ({ getContext: () => null }) });
  assert.equal(unavailable.applied, false);
  assert.match(unavailable.reason, /unavailable/);
  const enormous = refineMaskGPU({}, { width: 10000, height: 10000 }, {
    createCanvas: () => assert.fail("must reject before GPU allocation"),
  });
  assert.equal(enormous.applied, false);
  assert.match(enormous.reason, /budget/);
});

test("WebGL guided filter schedules seven full-resolution float passes and releases all resources", () => {
  const passes = [];
  const allocations = [];
  const deleted = [];
  const shaders = [];
  let active = 0;
  const units = [];
  let current;
  let target = null;
  let copied = false;
  let lost = false;
  const gl = {
    MAX_TEXTURE_SIZE: 1, FRAMEBUFFER_COMPLETE: 2, NO_ERROR: 0, TEXTURE0: 100,
    VERTEX_SHADER: 3, FRAGMENT_SHADER: 4, RGBA32F: 5, RGBA8: 6,
    getExtension: (name) => name === "EXT_color_buffer_float" ? {}
      : name === "WEBGL_lose_context" ? { loseContext: () => { lost = true; } } : null,
    getParameter: () => 4096,
    viewport: (x, y, w, h) => assert.deepEqual([x, y, w, h], [0, 0, 11, 7]),
    disable() {}, pixelStorei() {},
    createShader: (type) => { const shader = { type }; shaders.push(shader); return shader; },
    shaderSource: (shader, source) => { shader.source = source; },
    compileShader() {}, getShaderParameter: () => true,
    createProgram: () => ({ uniforms: {}, shaders: [] }),
    attachShader: (program, shader) => program.shaders.push(shader),
    linkProgram() {}, getProgramParameter: () => true,
    createTexture: () => ({}),
    bindTexture: (_, texture) => { units[active] = texture; },
    texParameteri() {},
    texImage2D: (...args) => allocations.push(args),
    createFramebuffer: () => ({}),
    useProgram: (program) => { current = program; },
    bindFramebuffer: (_, framebuffer) => { if (!framebuffer) target = null; },
    framebufferTexture2D: (a, b, c, texture) => { target = texture; },
    checkFramebufferStatus: () => 2,
    activeTexture: (unit) => { active = unit - 100; },
    getUniformLocation: (program, name) => ({ program, name }),
    uniform1i: ({ program, name }, unit) => { program.uniforms[name] = units[unit]; },
    uniform2f: ({ program, name }, x, y) => { program.uniforms[name] = [x, y]; },
    drawArrays() {
      assert.notEqual(current.uniforms.source, target, "must not sample the active float target");
      passes.push({ shader: current.shaders.find((shader) => shader.type === 4).source,
        uniforms: { ...current.uniforms }, target });
    },
    isContextLost: () => false, getError: () => 0,
    deleteTexture: (texture) => deleted.push(texture),
    deleteShader() {}, deleteProgram() {}, deleteFramebuffer() {},
  };
  const surface = { getContext: () => gl };
  const photo = { width: 11, height: 7 };
  const mask = { ...photo, getContext: () => ({ drawImage: (source) => {
    assert.equal(source, surface);
    copied = true;
  } }) };
  assert.deepEqual(refineMaskGPU(photo, mask, { createCanvas: () => surface }), { applied: true });
  assert.equal(passes.length, 7);
  assert.match(passes[0].shader, /i\*i, i\*p/);
  assert.match(passes[3].shader, /v\.w - v\.x\*v\.y/);
  assert.match(passes[6].shader, /ab\.x\*i \+ ab\.y/);
  for (const index of [1, 4]) assert.deepEqual(passes[index].uniforms.axis, [1, 0]);
  for (const index of [2, 5]) assert.deepEqual(passes[index].uniforms.axis, [0, 1]);
  assert.equal(passes[6].target, null);
  assert.equal(allocations.filter((args) => args[2] === gl.RGBA32F).length, 2);
  for (const args of allocations.filter((args) => args[2] === gl.RGBA32F)) {
    assert.equal(args[3], 11);
    assert.equal(args[4], 7);
  }
  assert.equal(deleted.length, 4);
  assert.equal(copied, true);
  assert.equal(lost, true);
});

test("guided refinement runs at full source resolution before trim for both model passes", async () => {
  const calls = [];
  const photo = new Canvas(11, 7, 128);
  let upgraded;
  const done = new Promise((resolve) => { upgraded = resolve; });
  const segmenter = createSegmenter({
    createCanvas: (w, h) => new Canvas(w, h),
    runtimeLoader: async () => runtime(),
    modelLoader: async (name) => name,
    refineMask(guide, mask) {
      assert.equal(guide, photo);
      assert.equal(mask.width, photo.width);
      assert.equal(mask.height, photo.height);
      // A low center value must spread through the following erosion.
      mask.data[(3 * 11 + 5) * 4] = 0;
      calls.push(mask);
      return { applied: true };
    },
  });
  const mask = await segmenter.segmentPhoto(photo, { trim: 1, onUpgrade: upgraded });
  const full = await done;
  assert.equal(calls.length, 2);
  assert.equal(mask.data[(3 * 11 + 4) * 4], 0);
  assert.equal(full.data[(3 * 11 + 6) * 4], 0);
});

test("rejects huge/invalid photos and reports model load failures without fabricated masks", async () => {
  const segmenter = createSegmenter({
    createCanvas: (w, h) => new Canvas(w, h),
    runtimeLoader: async () => runtime(),
    modelLoader: async () => { throw new Error("Model installation missing"); },
  });
  await assert.rejects(segmenter.segmentPhoto({ width: 10000, height: 10000 }), /Resize it in the caller/);
  await assert.rejects(segmenter.segmentPhoto(null), /nonempty/);
  await assert.rejects(segmenter.segmentPhoto(new Canvas(2, 2), { trim: -1 }), /trim/);
  await assert.rejects(segmenter.segmentPhoto(new Canvas(2, 2)), /Model installation missing/);
});
