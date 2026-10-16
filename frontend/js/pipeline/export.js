export const EXPORT_METADATA = Object.freeze({
  format: "Baseline JPEG, 4:4:4, quality 95, JFIF 300 dpi",
  memory: "Bounded RGBA stripes and one renderer tile; compressed Blob bytes remain in memory",
  rendererContract: "Return a Canvas2D-readable canvas of (tileWidth+2*margin) × (tileHeight+2*margin). Core starts at (margin,margin); x/y are document-space core coordinates, including at document edges.",
});

const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
const QUANT_Y = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const QUANT_C = [
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];
const BASIS = Float64Array.from({ length: 64 }, (_, i) => {
  const u = i >> 3;
  return (u === 0 ? Math.SQRT1_2 : 1) * Math.cos((2 * (i & 7) + 1) * u * Math.PI / 16) / 2;
});

// A deliberately simple, valid canonical Huffman table: 12 DC codes of length 4,
// and 162 AC codes of length 8. No code is all ones (reserved for JPEG padding).
const AC_SYMBOLS = [0, 0xf0];
for (let run = 0; run < 16; run++) {
  for (let size = 1; size <= 10; size++) AC_SYMBOLS.push(run * 16 + size);
}
const AC_CODES = new Map(AC_SYMBOLS.map((symbol, code) => [symbol, code]));

class JPEGEncoder {
  constructor(width, height) {
    this.parts = [];
    this.buffer = new Uint8Array(65536);
    this.offset = 0;
    this.bits = 0;
    this.bitCount = 0;
    this.dc = [0, 0, 0];
    this.quant = [QUANT_Y, QUANT_C].map((table) => table.map((value) => Math.max(1, Math.floor((value * 10 + 50) / 100))));
    this.block = new Float64Array(64);
    this.temp = new Float64Array(64);
    this.coefficients = new Int32Array(64);
    this.marker(0xd8);
    this.marker(0xe0);
    this.word(16);
    this.bytes([74, 70, 73, 70, 0, 1, 1, 1]); // JFIF 1.01, density in dots/inch.
    this.word(300);
    this.word(300);
    this.bytes([0, 0]);
    this.marker(0xdb);
    this.word(132);
    for (let table = 0; table < 2; table++) {
      this.byte(table);
      for (const index of ZIGZAG) this.byte(this.quant[table][index]);
    }
    this.marker(0xc0);
    this.word(17);
    this.byte(8);
    this.word(height);
    this.word(width);
    this.bytes([3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1]);
    this.marker(0xc4);
    this.word(2 + 17 + 12 + 17 + AC_SYMBOLS.length);
    this.byte(0);
    for (let size = 1; size <= 16; size++) this.byte(size === 4 ? 12 : 0);
    for (let value = 0; value < 12; value++) this.byte(value);
    this.byte(0x10);
    for (let size = 1; size <= 16; size++) this.byte(size === 8 ? AC_SYMBOLS.length : 0);
    this.bytes(AC_SYMBOLS);
    this.marker(0xda);
    this.word(12);
    this.bytes([3, 1, 0, 2, 0, 3, 0, 0, 63, 0]);
  }

  byte(value) {
    this.buffer[this.offset++] = value;
    if (this.offset === this.buffer.length) {
      this.parts.push(this.buffer);
      this.buffer = new Uint8Array(65536);
      this.offset = 0;
    }
  }
  bytes(values) { for (const value of values) this.byte(value); }
  word(value) { this.byte(value >> 8); this.byte(value & 255); }
  marker(value) { this.byte(255); this.byte(value); }

  writeBits(value, count) {
    this.bits = (this.bits << count) | value;
    this.bitCount += count;
    while (this.bitCount >= 8) {
      this.bitCount -= 8;
      const byte = (this.bits >>> this.bitCount) & 255;
      this.byte(byte);
      if (byte === 255) this.byte(0);
    }
    this.bits &= (1 << this.bitCount) - 1;
  }

  amplitude(value) {
    const size = value ? Math.floor(Math.log2(Math.abs(value))) + 1 : 0;
    return [size, value < 0 ? value + (1 << size) - 1 : value];
  }

  encodeBlock(component) {
    const { block, temp, coefficients } = this;
    const quant = this.quant[component ? 1 : 0];
    for (let y = 0; y < 8; y++) {
      for (let u = 0; u < 8; u++) {
        let sum = 0;
        for (let x = 0; x < 8; x++) sum += block[y * 8 + x] * BASIS[u * 8 + x];
        temp[y * 8 + u] = sum;
      }
    }
    for (let v = 0; v < 8; v++) {
      for (let u = 0; u < 8; u++) {
        let sum = 0;
        for (let y = 0; y < 8; y++) sum += temp[y * 8 + u] * BASIS[v * 8 + y];
        coefficients[v * 8 + u] = Math.round(sum / quant[v * 8 + u]);
      }
    }
    const [size, value] = this.amplitude(coefficients[0] - this.dc[component]);
    this.dc[component] = coefficients[0];
    this.writeBits(size, 4);
    this.writeBits(value, size);
    let last = 63;
    while (last > 0 && !coefficients[ZIGZAG[last]]) last--;
    let zeros = 0;
    for (let i = 1; i <= last; i++) {
      const coefficient = coefficients[ZIGZAG[i]];
      if (!coefficient) { zeros++; continue; }
      while (zeros >= 16) {
        this.writeBits(AC_CODES.get(0xf0), 8);
        zeros -= 16;
      }
      const [acSize, acValue] = this.amplitude(coefficient);
      this.writeBits(AC_CODES.get(zeros * 16 + acSize), 8);
      this.writeBits(acValue, acSize);
      zeros = 0;
    }
    if (last !== 63) this.writeBits(AC_CODES.get(0), 8);
  }

  rows(rgba, width, height) {
    for (let by = 0; by < height; by += 8) {
      for (let bx = 0; bx < width; bx += 8) {
        for (let c = 0; c < 3; c++) {
          for (let y = 0; y < 8; y++) {
            for (let x = 0; x < 8; x++) {
              const offset = (Math.min(by + y, height - 1) * width + Math.min(bx + x, width - 1)) * 4;
              const alpha = rgba[offset + 3] / 255;
              const r = rgba[offset] * alpha + 255 * (1 - alpha);
              const g = rgba[offset + 1] * alpha + 255 * (1 - alpha);
              const b = rgba[offset + 2] * alpha + 255 * (1 - alpha);
              this.block[y * 8 + x] = c === 0 ? 0.299 * r + 0.587 * g + 0.114 * b - 128
                : c === 1 ? -0.168736 * r - 0.331264 * g + 0.5 * b
                  : 0.5 * r - 0.418688 * g - 0.081312 * b;
            }
          }
          this.encodeBlock(c);
        }
      }
    }
  }

  finish() {
    if (this.bitCount) this.writeBits((1 << (8 - this.bitCount)) - 1, 8 - this.bitCount);
    this.marker(0xd9);
    this.parts.push(this.buffer.slice(0, this.offset));
    return new Blob(this.parts, { type: "image/jpeg" });
  }
}

/**
 * Export WatercolorDocument.outputWidth/outputHeight, replayed by renderTile.
 * Tiles include overlap on ALL four sides, even outside the document bounds.
 * onBlob is awaited once; the same Blob is also returned. No download side effect.
 */
export async function exportJPG({ document, renderTile, onProgress = () => {},
  onBlob = () => {}, tileSize = 512, margin = 32 } = {}) {
  const width = document?.outputWidth;
  const height = document?.outputHeight;
  if (![width, height].every((value) => Number.isSafeInteger(value) && value > 0 && value <= 65535)) {
    throw new RangeError("JPEG dimensions must be integers between 1 and 65535 pixels.");
  }
  if (typeof renderTile !== "function") throw new TypeError("exportJPG requires renderTile.");
  if (!Number.isSafeInteger(tileSize) || tileSize < 8 || tileSize > 2048 ||
      !Number.isSafeInteger(margin) || margin < 0 || margin > 512) {
    throw new RangeError("tileSize must be 8–2048 and margin must be 0–512 integer pixels.");
  }
  const encoder = new JPEGEncoder(width, height);
  // Never allocate a document-sized canvas. Every non-final stripe is MCU-aligned.
  const stripeHeight = Math.min(128, Math.max(8, Math.floor(8 * 1024 * 1024 / (width * 4) / 8) * 8));
  onProgress({ stage: "export", progress: 0, message: "Rendering overlapping JPEG tiles" });
  for (let y = 0; y < height; y += stripeHeight) {
    const rows = Math.min(stripeHeight, height - y);
    const stripe = new Uint8ClampedArray(width * rows * 4);
    for (let x = 0; x < width; x += tileSize) {
      const tileWidth = Math.min(tileSize, width - x);
      // tileSize can be smaller than the stripe; split vertically as well.
      for (let dy = 0; dy < rows; dy += tileSize) {
        const tileHeight = Math.min(tileSize, rows - dy);
        const tile = await renderTile({ width, height, x, y: y + dy, tileWidth, tileHeight, margin });
        if (tile?.width !== tileWidth + 2 * margin || tile?.height !== tileHeight + 2 * margin) {
          throw new Error("renderTile must return the core tile plus margin on all four sides.");
        }
        let pixels;
        try {
          const ctx = tile.getContext("2d", { willReadFrequently: true });
          if (!ctx) throw new Error("Canvas2D readback is unavailable");
          pixels = ctx.getImageData(margin, margin, tileWidth, tileHeight).data;
        } catch (cause) {
          throw new Error("Cannot read export tile. Return a Canvas2D canvas; use CORS-enabled images and check canvas memory.", { cause });
        }
        for (let row = 0; row < tileHeight; row++) {
          stripe.set(pixels.subarray(row * tileWidth * 4, (row + 1) * tileWidth * 4),
            ((dy + row) * width + x) * 4);
        }
      }
    }
    encoder.rows(stripe, width, rows);
    onProgress({ stage: "export", progress: (y + rows) / height, message: `Encoded ${y + rows} of ${height} rows` });
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const blob = encoder.finish();
  await onBlob(blob);
  onProgress({ stage: "complete", progress: 1, message: "300 dpi JPEG ready" });
  return blob;
}
