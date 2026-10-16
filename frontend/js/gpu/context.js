export class TexturePool {
  constructor(gl, budgetBytes = 250 * 1024 * 1024) {
    this.gl = gl;
    this.budgetBytes = budgetBytes;
    this.textures = new Map();
    this.bytesUsed = 0;
  }

  create(width, height, { format = this.gl.RGBA, type = this.gl.UNSIGNED_BYTE,
    minFilter = this.gl.LINEAR, magFilter = this.gl.LINEAR } = {}) {
    const channels = format === this.gl.RGBA ? 4 : 1;
    const bytesPerChannel = type === this.gl.FLOAT ? 4 : 1;
    const bytes = width * height * channels * bytesPerChannel;
    if (width <= 0 || height <= 0 || this.bytesUsed + bytes > this.budgetBytes) {
      throw new RangeError("Requested GPU texture exceeds the available memory budget.");
    }
    const texture = this.gl.createTexture();
    if (!texture) throw new Error("Unable to allocate a WebGL texture.");
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, minFilter);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, magFilter);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, type, null);
    this.textures.set(texture, bytes);
    this.bytesUsed += bytes;
    return texture;
  }

  release(texture) {
    const bytes = this.textures.get(texture);
    if (bytes === undefined) return;
    this.gl.deleteTexture(texture);
    this.textures.delete(texture);
    this.bytesUsed -= bytes;
  }

  dispose() {
    for (const texture of this.textures.keys()) this.gl.deleteTexture(texture);
    this.textures.clear();
    this.bytesUsed = 0;
  }
}

export function createWebGL2Context(canvas, options = {}) {
  const gl = canvas?.getContext("webgl2", {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    ...options,
  });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");
  return { gl, texturePool: new TexturePool(gl) };
}
