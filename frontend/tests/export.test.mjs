import assert from "node:assert/strict";
import test from "node:test";
import { exportJPG, EXPORT_METADATA } from "../js/pipeline/export.js";

function renderer(calls, color = [210, 80, 35, 255]) {
  return async (options) => {
    calls.push(options);
    const { tileWidth, tileHeight, margin } = options;
    return {
      width: tileWidth + margin * 2, height: tileHeight + margin * 2,
      getContext: () => ({
        getImageData(x, y, w, h) {
          assert.equal(x, margin);
          assert.equal(y, margin);
          assert.equal(w, tileWidth);
          assert.equal(h, tileHeight);
          return { data: Uint8ClampedArray.from({ length: w * h * 4 }, (_, i) => {
            if (typeof color !== "function") return color[i % 4];
            const pixel = Math.floor(i / 4);
            return color(options.x + pixel % w, options.y + Math.floor(pixel / w))[i % 4];
          }) };
        },
      }),
    };
  };
}

function headers(bytes) {
  const segments = new Map();
  let offset = 2;
  while (offset < bytes.length) {
    assert.equal(bytes[offset++], 255);
    const marker = bytes[offset++];
    const length = bytes[offset] * 256 + bytes[offset + 1];
    const payload = bytes.slice(offset + 2, offset + length);
    segments.set(marker, payload);
    offset += length;
    if (marker === 0xda) return { segments, entropy: bytes.slice(offset, -2) };
  }
  assert.fail("missing JPEG scan");
}

// Independent baseline decoder for entropy, quantizers and sampled inverse DCT.
function decodeDC(bytes) {
  const { segments, entropy } = headers(bytes);
  const dht = segments.get(0xc4);
  const tables = new Map();
  for (let offset = 0; offset < dht.length;) {
    const id = dht[offset++];
    const counts = dht.slice(offset, offset + 16);
    offset += 16;
    const codes = new Map();
    let code = 0;
    for (let length = 1; length <= 16; length++) {
      for (let i = 0; i < counts[length - 1]; i++) codes.set(`${length}:${code++}`, dht[offset++]);
      code *= 2;
    }
    tables.set(id, codes);
  }
  const unstuffed = [];
  for (let i = 0; i < entropy.length; i++) {
    unstuffed.push(entropy[i]);
    if (entropy[i] === 255) assert.equal(entropy[++i], 0);
  }
  let bit = 0;
  function read(n) {
    let value = 0;
    for (let i = 0; i < n; i++, bit++) {
      assert.ok(bit < unstuffed.length * 8, "truncated entropy");
      value = value * 2 + ((unstuffed[bit >> 3] >> (7 - bit % 8)) & 1);
    }
    return value;
  }
  function symbol(id) {
    let code = 0;
    for (let length = 1; length <= 16; length++) {
      code = code * 2 + read(1);
      const value = tables.get(id).get(`${length}:${code}`);
      if (value !== undefined) return value;
    }
    assert.fail("invalid Huffman code");
  }
  function amplitude(size) {
    const value = read(size);
    return size && value < 2 ** (size - 1) ? value - 2 ** size + 1 : value;
  }
  const frame = segments.get(0xc0);
  const height = frame[1] * 256 + frame[2];
  const width = frame[3] * 256 + frame[4];
  const last = [0, 0, 0];
  const output = [];
  const blocks = [];
  const zigzag = [];
  for (let diagonal = 0; diagonal <= 14; diagonal++) {
    const indices = [];
    for (let y = 0; y < 8; y++) {
      const x = diagonal - y;
      if (x >= 0 && x < 8) indices.push(y * 8 + x);
    }
    zigzag.push(...(diagonal % 2 ? indices : indices.reverse()));
  }
  for (let block = 0; block < Math.ceil(width / 8) * Math.ceil(height / 8); block++) {
    const channels = [];
    for (let c = 0; c < 3; c++) {
      last[c] += amplitude(symbol(0));
      const coefficients = new Float64Array(64);
      coefficients[0] = last[c];
      let index = 1;
      while (index < 64) {
        const ac = symbol(0x10);
        if (!ac) break;
        if (ac === 240) index += 16;
        else {
          index += ac >> 4;
          coefficients[zigzag[index++]] = amplitude(ac & 15);
        }
        assert.ok(index <= 64);
      }
      channels.push(coefficients);
    }
    output.push([...last]);
    blocks.push(channels);
  }
  function pixel(x, y) {
    const block = blocks[Math.floor(y / 8) * Math.ceil(width / 8) + Math.floor(x / 8)];
    const quantizers = segments.get(0xdb);
    const samples = block.map((coefficients, c) => {
      let sum = 0;
      for (let i = 0; i < 64; i++) {
        const natural = zigzag[i];
        const u = natural % 8;
        const v = Math.floor(natural / 8);
        sum += coefficients[natural] * quantizers[(c ? 65 : 0) + 1 + i] *
          (u ? 1 : Math.SQRT1_2) * (v ? 1 : Math.SQRT1_2) *
          Math.cos((2 * (x % 8) + 1) * u * Math.PI / 16) *
          Math.cos((2 * (y % 8) + 1) * v * Math.PI / 16) / 4;
      }
      return sum;
    });
    const [luma, cb, cr] = samples;
    return [luma + 128 + 1.402 * cr, luma + 128 - 0.344136 * cb - 0.714136 * cr,
      luma + 128 + 1.772 * cb].map((value) => Math.max(0, Math.min(255, Math.round(value))));
  }
  return { output, segments, pixel };
}

test("exports a real baseline quality95 JPEG with 300dpi JFIF and decodable DC colors", async () => {
  const calls = [];
  const events = [];
  let delivered;
  const blob = await exportJPG({
    document: { outputWidth: 19, outputHeight: 13 },
    tileSize: 8, margin: 3, renderTile: renderer(calls),
    onProgress: (event) => events.push(event), onBlob: async (value) => { delivered = value; },
  });
  assert.equal(delivered, blob);
  assert.equal(blob.type, "image/jpeg");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  assert.deepEqual([...bytes.slice(0, 2)], [255, 216]);
  assert.deepEqual([...bytes.slice(-2)], [255, 217]);
  const { output, segments } = decodeDC(bytes);
  const jfif = segments.get(0xe0);
  assert.deepEqual([...jfif], [74, 70, 73, 70, 0, 1, 1, 1, 1, 44, 1, 44, 0, 0]);
  assert.equal(segments.get(0xdb)[1], 2); // Quality 95: luminance quantizer 16 -> 2.
  assert.equal(output.length, 6);
  // Constant RGB [210,80,35] -> Y-128 ~= -14.26, Cb ~= -44.44, Cr ~= 68.66.
  for (const [y, cb, cr] of output) {
    assert.ok(Math.abs(y * 2 / 8 + 14.26) < 0.3);
    assert.ok(Math.abs(cb * 2 / 8 + 44.44) < 0.3);
    assert.ok(Math.abs(cr * 2 / 8 - 68.66) < 0.3);
  }
  assert.equal(calls.length, 6);
  assert.deepEqual(calls.at(-1), { width: 19, height: 13, x: 16, y: 8, tileWidth: 3, tileHeight: 5, margin: 3 });
  assert.equal(events.at(-1).stage, "complete");
});

test("stripe encoding covers odd final rows exactly and composites transparency on white", async () => {
  const calls = [];
  const blob = await exportJPG({
    document: { outputWidth: 9, outputHeight: 131 },
    renderTile: renderer(calls, [0, 0, 0, 0]), margin: 2,
  });
  assert.deepEqual(calls.map(({ y, tileHeight }) => [y, tileHeight]), [[0, 128], [128, 3]]);
  const { output } = decodeDC(new Uint8Array(await blob.arrayBuffer()));
  assert.equal(output.length, 2 * 17);
  for (const dc of output) assert.deepEqual(dc, [508, 0, 0]);
});

test("very wide output still requests bounded renderer tiles, never a full document canvas", async () => {
  const calls = [];
  await exportJPG({ document: { outputWidth: 10001, outputHeight: 1 }, renderTile: renderer(calls) });
  assert.equal(calls.length, 20);
  assert.ok(calls.every(({ tileWidth, tileHeight }) => tileWidth <= 512 && tileHeight === 1));
  assert.match(EXPORT_METADATA.memory, /compressed Blob bytes remain/);
});

test("nonconstant RGB survives AC encoding, tile seams, stripe boundaries and final block padding", async () => {
  const color = (x, y) => [80 + x * 5, 40 + y, 25 + x * 2 + y, 255];
  const blob = await exportJPG({
    document: { outputWidth: 19, outputHeight: 131 },
    tileSize: 8, margin: 5, renderTile: renderer([], color),
  });
  const decoded = decodeDC(new Uint8Array(await blob.arrayBuffer()));
  for (const [x, y] of [[0, 0], [7, 7], [8, 8], [18, 12], [0, 127], [12, 128], [18, 130]]) {
    const actual = decoded.pixel(x, y);
    const expected = color(x, y);
    actual.forEach((value, c) => assert.ok(Math.abs(value - expected[c]) <= 4,
      `pixel (${x},${y}) channel ${c}: ${value} vs ${expected[c]}`));
  }
});

test("rejects invalid dimensions, missing overlap, unreadable tiles and propagates renderer failures", async () => {
  await assert.rejects(exportJPG({ document: { outputWidth: 65536, outputHeight: 1 } }), /65535/);
  await assert.rejects(exportJPG({ document: { outputWidth: 1.2, outputHeight: 1 } }), /integers/);
  const document = { outputWidth: 3, outputHeight: 2 };
  await assert.rejects(exportJPG({ document }), /renderTile/);
  await assert.rejects(exportJPG({ document, renderTile() {}, margin: -1 }), /margin/);
  await assert.rejects(exportJPG({ document, renderTile: async () => ({ width: 3, height: 2 }) }), /margin/);
  await assert.rejects(exportJPG({
    document, margin: 0,
    renderTile: async () => ({ width: 3, height: 2, getContext: () => null }),
  }), /Canvas2D/);
  await assert.rejects(exportJPG({
    document, renderTile: async () => { throw new Error("Renderer unavailable"); },
  }), /Renderer unavailable/);
});
