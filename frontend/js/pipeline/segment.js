const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const MODEL_NAMES = ["u2netp", "u2net"];
const MAX_PIXELS = 16_777_216;

export const SEGMENTATION_APPROXIMATIONS = Object.freeze({
  inputResampling: "Canvas2D high-quality resampling, not Pillow Lanczos",
  outputResampling: "Canvas2D high-quality resampling, not Pillow Lanczos",
  edgeRefinement: "Full-resolution grayscale-guide WebGL2 guided filter, radius 8, epsilon 0.001, then CPU minimum-filter erosion; refinement explicitly reported as skipped if float render targets or GPU budget are unavailable",
  maskFormat: "Opaque grayscale RGB (0 background, 255 foreground); not alpha",
  trimUnits: "Pixels at the supplied photo resolution",
});

function canvas(width, height) {
  const result = typeof OffscreenCanvas === "function"
    ? new OffscreenCanvas(width, height)
    : globalThis.document?.createElement("canvas");
  if (!result) throw new Error("Segmentation needs a browser with Canvas2D support.");
  result.width = width;
  result.height = height;
  return result;
}

function context(surface) {
  const result = surface.getContext("2d", { willReadFrequently: true });
  if (!result) throw new Error("Canvas2D allocation failed. Resize the photo and retry.");
  return result;
}

// rembg BaseSession.normalize: RGB / global maximum, then channel mean/std, NCHW.
export function normalizeInput(rgba) {
  const pixels = rgba.length / 4;
  const data = new Float32Array(pixels * 3);
  let maximum = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    maximum = Math.max(maximum, rgba[i], rgba[i + 1], rgba[i + 2]);
  }
  const means = [0.485, 0.456, 0.406];
  const stds = [0.229, 0.224, 0.225];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < pixels; i++) {
      data[c * pixels + i] = ((maximum ? rgba[i * 4 + c] / maximum : 0) - means[c]) / stds[c];
    }
  }
  return data;
}

export function normalizeOutput(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new Error("The segmentation model returned non-finite values.");
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const result = new Uint8ClampedArray(values.length);
  for (let i = 0; i < values.length; i++) {
    result[i] = max > min ? Math.floor((values[i] - min) / (max - min) * 255) : 0;
  }
  return result;
}

// Sliding minima keep trimming linear even for wide radii. Edges are clamped.
export function erodeMask(values, width, height, radius) {
  if (!radius) return values;
  const temp = new Uint8ClampedArray(values.length);
  const result = new Uint8ClampedArray(values.length);
  const queue = new Int32Array(Math.max(width, height));
  function pass(source, target, length, lines, stride, lineStride) {
    for (let line = 0; line < lines; line++) {
      const base = line * lineStride;
      let head = 0;
      let tail = 0;
      let next = 0;
      for (let i = 0; i < length; i++) {
        const end = Math.min(length - 1, i + radius);
        while (next <= end) {
          while (tail > head && source[base + queue[tail - 1] * stride] >= source[base + next * stride]) tail--;
          queue[tail++] = next++;
        }
        while (queue[head] < i - radius) head++;
        target[base + i * stride] = source[base + queue[head] * stride];
      }
    }
  }
  pass(values, temp, width, height, 1, width);
  pass(temp, result, height, width, width, 1);
  return result;
}

/**
 * He et al. grayscale guided filter at source resolution, not a bilateral blur.
 * Two reusable float targets keep storage within a 192 MiB working budget.
 */
export function refineMaskGPU(photo, mask, { createCanvas = canvas } = {}) {
  const { width, height } = mask;
  if (width * height * 48 > 192 * 1024 * 1024) {
    return { applied: false, reason: "Guided-filter GPU budget exceeded; resize the source photo to at most 2048 pixels on its longest edge." };
  }
  let gl;
  const textures = [];
  const programs = [];
  const shaders = [];
  let framebuffer;
  const vertex = `#version 300 es
    void main() {
      vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
      gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }`;
  const preamble = `#version 300 es
    precision highp float;
    uniform vec2 size;
    out vec4 color;
    vec2 uv() { return gl_FragCoord.xy / size; }`;
  try {
    const surface = createCanvas(width, height);
    gl = surface.getContext("webgl2", {
      alpha: false, depth: false, stencil: false, antialias: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true,
    });
    if (!gl?.getExtension?.("EXT_color_buffer_float")) {
      return { applied: false, reason: "WebGL2 floating-point render targets are unavailable on this device." };
    }
    if (Math.max(width, height) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) {
      return { applied: false, reason: "Photo exceeds the GPU texture limit; resize it before segmentation." };
    }
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    function compile(type, source) {
      const shader = gl.createShader(type);
      if (!shader) throw new Error("Cannot allocate guided-filter shader.");
      shaders.push(shader);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    }
    const vs = compile(gl.VERTEX_SHADER, vertex);
    function program(fragment) {
      const result = gl.createProgram();
      if (!result) throw new Error("Cannot allocate guided-filter program.");
      programs.push(result);
      gl.attachShader(result, vs);
      gl.attachShader(result, compile(gl.FRAGMENT_SHADER, preamble + fragment));
      gl.linkProgram(result);
      if (!gl.getProgramParameter(result, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(result));
      return result;
    }
    const moments = program(`
      uniform sampler2D guide;
      uniform sampler2D mask;
      void main() {
        float i = dot(texture(guide, uv()).rgb, vec3(0.299, 0.587, 0.114));
        float p = texture(mask, uv()).r;
        color = vec4(i, p, i*i, i*p);
      }`);
    const box = program(`
      uniform sampler2D source;
      uniform vec2 axis;
      void main() {
        vec4 sum = vec4(0.0);
        for (int offset = -8; offset <= 8; offset++)
          sum += texture(source, uv() + axis * float(offset) / size);
        color = sum / 17.0;
      }`);
    const coefficients = program(`
      uniform sampler2D source;
      void main() {
        vec4 v = texture(source, uv());
        float a = (v.w - v.x*v.y) / (max(0.0, v.z - v.x*v.x) + 0.001);
        color = vec4(a, v.y - a*v.x, 0.0, 1.0);
      }`);
    const reconstruct = program(`
      uniform sampler2D guide;
      uniform sampler2D source;
      void main() {
        float i = dot(texture(guide, uv()).rgb, vec3(0.299, 0.587, 0.114));
        vec2 ab = texture(source, uv()).rg;
        float q = clamp(ab.x*i + ab.y, 0.0, 1.0);
        color = vec4(q, q, q, 1.0);
      }`);
    function texture(image) {
      const result = gl.createTexture();
      if (!result) throw new Error("Cannot allocate guided-filter texture.");
      textures.push(result);
      gl.bindTexture(gl.TEXTURE_2D, result);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (image) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      return result;
    }
    // UNPACK_FLIP_Y_WEBGL is ignored for ImageBitmap uploads. Convert that case
    // through Canvas2D so both guide and mask use identical bottom-left UVs.
    let guideSource = photo;
    if (typeof ImageBitmap === "function" && photo instanceof ImageBitmap) {
      guideSource = createCanvas(width, height);
      context(guideSource).drawImage(photo, 0, 0);
    }
    const guideTexture = texture(guideSource);
    const maskTexture = texture(mask);
    const a = texture();
    const b = texture();
    framebuffer = gl.createFramebuffer();
    if (!framebuffer) throw new Error("Cannot allocate guided-filter framebuffer.");
    function draw(shader, target, sources, axis = [0, 0]) {
      gl.useProgram(shader);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target ? framebuffer : null);
      if (target) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
          throw new Error("Incomplete guided-filter float framebuffer.");
        }
      }
      let unit = 0;
      for (const [name, input] of Object.entries(sources)) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, input);
        gl.uniform1i(gl.getUniformLocation(shader, name), unit++);
      }
      gl.uniform2f(gl.getUniformLocation(shader, "size"), width, height);
      gl.uniform2f(gl.getUniformLocation(shader, "axis"), ...axis);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    draw(moments, a, { guide: guideTexture, mask: maskTexture });
    draw(box, b, { source: a }, [1, 0]);
    draw(box, a, { source: b }, [0, 1]);
    draw(coefficients, b, { source: a });
    draw(box, a, { source: b }, [1, 0]);
    draw(box, b, { source: a }, [0, 1]);
    draw(reconstruct, null, { guide: guideTexture, source: b });
    if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error("GPU allocation or guided-filter draw failed.");
    context(mask).drawImage(surface, 0, 0);
    return { applied: true };
  } catch (error) {
    return { applied: false, reason: `Guided filter unavailable: ${error.message}` };
  } finally {
    if (gl?.deleteTexture) {
      for (const texture of textures) gl.deleteTexture(texture);
      for (const shader of shaders) gl.deleteShader(shader);
      for (const program of programs) gl.deleteProgram(program);
      if (framebuffer) gl.deleteFramebuffer(framebuffer);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    }
  }
}

async function loadRuntime() {
  try {
    const ort = await import(`${ORT_BASE}ort.webgpu.min.mjs`);
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.min(4, globalThis.navigator?.hardwareConcurrency || 1) : 1;
    return ort;
  } catch (cause) {
    throw new Error("Cannot load ONNX Runtime. Allow cdn.jsdelivr.net in your network/CSP, then retry.", { cause });
  }
}

async function fetchModel(name, onProgress) {
  const url = new URL(`../../models/${name}.onnx`, import.meta.url).href;
  let cache;
  try {
    cache = await globalThis.caches?.open("watercolor-u2net-v1");
    const cached = await cache?.match(url);
    if (cached?.ok) return await cached.arrayBuffer();
  } catch {
    // Storage can be denied in private mode; inference still works without caching.
  }
  onProgress({ stage: "download", progress: 0, message: `Downloading ${name}.onnx` });
  let response;
  try {
    response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (cause) {
    throw new Error(`Cannot fetch ${name}.onnx. Serve the model at ${url}; check your connection and model installation.`, { cause });
  }
  const total = Number(response.headers.get("content-length")) || 0;
  let bytes;
  if (response.body?.getReader) {
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        onProgress({
          stage: "download",
          progress: total ? Math.min(0.99, received / total) : 0,
          message: `Downloading ${name}.onnx: ${(received / 1048576).toFixed(1)} MiB${total ? ` of ${(total / 1048576).toFixed(1)} MiB` : " (size unknown)"}`,
        });
      }
    } catch (cause) {
      await reader.cancel().catch(() => {});
      throw new Error(`Download of ${name}.onnx was interrupted. Check your connection and retry.`, { cause });
    } finally {
      reader.releaseLock();
    }
    const joined = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      joined.set(chunk, offset);
      offset += chunk.byteLength;
    }
    bytes = joined.buffer;
  } else {
    bytes = await response.arrayBuffer();
  }
  if (cache) {
    try { await cache.put(url, new Response(bytes)); } catch { /* Quota must not prevent inference. */ }
  }
  onProgress({ stage: "download", progress: 1, message: `${name}.onnx downloaded` });
  return bytes;
}

// Factory also allows deterministic tests without loading remote code or models.
export function createSegmenter({ runtimeLoader = loadRuntime, modelLoader = fetchModel,
  createCanvas = canvas, refineMask = refineMaskGPU } = {}) {
  let runtimePromise;
  const sessions = new Map();
  const queues = new WeakMap();
  const runtime = () => runtimePromise ||= runtimeLoader().catch((error) => {
    runtimePromise = null;
    throw error;
  });

  function getSession(name, onProgress) {
    if (!sessions.has(name)) {
      const pending = (async () => {
        const ort = await runtime();
        const bytes = await modelLoader(name, onProgress);
        onProgress({ stage: "model", progress: 0, message: `Preparing ${name}` });
        let session;
        let gpu = false;
        if (globalThis.navigator?.gpu) {
          try {
            session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
            gpu = true;
          } catch { /* Some devices/operators cannot execute this model on WebGPU. */ }
        }
        if (!session) {
          try {
            session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
          } catch (cause) {
            throw new Error(`Cannot initialize ${name}. Check the ONNX model and allow the pinned Runtime WASM assets in your CSP.`, { cause });
          }
        }
        onProgress({ stage: "model", progress: 1, message: `${name} ready (${gpu ? "WebGPU" : "WASM"})` });
        return { session, gpu, bytes: gpu ? bytes : null, ort };
      })().catch((error) => {
        sessions.delete(name);
        throw error;
      });
      sessions.set(name, pending);
    }
    return sessions.get(name);
  }

  async function infer(model, input, photo, width, height, trim, onProgress) {
    // Serialize runs on a session; a WebGPU runtime failure is retried on WASM.
    const previous = queues.get(model) || Promise.resolve();
    const running = previous.catch(() => {}).then(async () => {
      let output;
      const run = async () => {
        const tensor = new model.ort.Tensor("float32", input, [1, 3, 320, 320]);
        try { return await model.session.run({ [model.session.inputNames[0]]: tensor }); }
        finally { tensor.dispose?.(); }
      };
      try {
        output = await run();
      } catch (cause) {
        if (!model.gpu) throw new Error("Segmentation inference failed. Resize the photo or reload the model and retry.", { cause });
        await model.session.release?.();
        try {
          model.session = await model.ort.InferenceSession.create(model.bytes, { executionProviders: ["wasm"] });
          model.bytes = null;
          model.gpu = false;
          output = await run();
        } catch (fallbackCause) {
          throw new Error("WebGPU and WASM inference failed. Allow Runtime WASM assets in your CSP, or reload and retry on a device with more memory.", { cause: fallbackCause });
        }
      }
      try {
        const prediction = output[model.session.outputNames[0]];
        const dims = prediction?.dims;
        if (!prediction || dims?.length !== 4 || dims[0] !== 1 || dims[1] !== 1 ||
            dims[2] !== 320 || dims[3] !== 320) {
          throw new Error("Unexpected u2net output: expected the first output to be [1,1,320,320].");
        }
        const low = createCanvas(320, 320);
        const lowContext = context(low);
        const image = lowContext.createImageData(320, 320);
        const normalized = normalizeOutput(prediction.data);
        for (let i = 0; i < normalized.length; i++) {
          image.data.fill(normalized[i], i * 4, i * 4 + 3);
          image.data[i * 4 + 3] = 255;
        }
        lowContext.putImageData(image, 0, 0);
        const mask = createCanvas(width, height);
        const maskContext = context(mask);
        maskContext.imageSmoothingEnabled = true;
        maskContext.imageSmoothingQuality = "high";
        maskContext.drawImage(low, 0, 0, width, height);
        const refinement = refineMask(photo, mask, { createCanvas });
        onProgress(refinement.applied
          ? { stage: "refine", progress: 1, message: "Full-resolution GPU guided filter complete" }
          : { stage: "refine-skipped", progress: 0, message: refinement.reason });
        if (trim) {
          const full = maskContext.getImageData(0, 0, width, height);
          const gray = new Uint8ClampedArray(width * height);
          for (let i = 0; i < gray.length; i++) gray[i] = full.data[i * 4];
          const eroded = erodeMask(gray, width, height, trim);
          for (let i = 0; i < gray.length; i++) full.data.fill(eroded[i], i * 4, i * 4 + 3);
          maskContext.putImageData(full, 0, 0);
        }
        return mask;
      } finally {
        for (const value of Object.values(output)) value.dispose?.();
      }
    });
    queues.set(model, running);
    try { return await running; }
    finally { if (queues.get(model) === running) queues.delete(model); }
  }

  async function warmModels(onProgress = () => {}) {
    await getSession(MODEL_NAMES[0], onProgress);
    await getSession(MODEL_NAMES[1], onProgress);
  }

  async function segmentPhoto(photo, { onProgress = () => {}, onUpgrade = () => {}, trim = 6 } = {}) {
    const { width, height } = photo || {};
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
      throw new TypeError("segmentPhoto requires a nonempty ImageBitmap or canvas.");
    }
    if (width * height > MAX_PIXELS) {
      throw new RangeError("Photo exceeds the 16-megapixel segmentation budget. Resize it in the caller before segmentPhoto.");
    }
    if (!Number.isSafeInteger(trim) || trim < 0) {
      throw new RangeError("trim must be a nonnegative integer pixel radius.");
    }
    const small = createCanvas(320, 320);
    const ctx = context(small);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    try { ctx.drawImage(photo, 0, 0, 320, 320); } catch (cause) {
      throw new Error("Cannot read the photo. Use an open ImageBitmap or a valid canvas.", { cause });
    }
    let input;
    try { input = normalizeInput(ctx.getImageData(0, 0, 320, 320).data); } catch (cause) {
      throw new Error("Cannot read photo pixels. Load remote photos with CORS permission before segmentation.", { cause });
    }
    const smallModel = await getSession("u2netp", onProgress);
    // Attach a handler immediately: background failures never become unhandled rejections.
    const largeModel = getSession("u2net", onProgress).then(
      (model) => ({ model }), (error) => ({ error }),
    );
    onProgress({ stage: "segment", progress: 0, message: "Segmenting with u2netp" });
    const mask = await infer(smallModel, input, photo, width, height, trim, onProgress);
    onProgress({ stage: "segment", progress: 1, message: "Initial mask ready" });
    void largeModel.then(async ({ model, error }) => {
      if (error) throw error;
      const upgraded = await infer(model, input, photo, width, height, trim, onProgress);
      await onUpgrade(upgraded);
      onProgress({ stage: "upgrade", progress: 1, message: "Full u2net mask ready" });
    }).catch((error) => {
      onProgress({ stage: "upgrade-error", progress: 0, message: `Initial mask retained. ${error.message}` });
    }).catch(() => {});
    return mask;
  }
  return { segmentPhoto, warmModels };
}

const defaultSegmenter = createSegmenter();
export const segmentPhoto = defaultSegmenter.segmentPhoto;
export const warmModels = defaultSegmenter.warmModels;
