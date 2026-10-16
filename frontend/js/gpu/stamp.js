import { STAMP_FRAGMENT_SHADER, STAMP_VERTEX_SHADER } from "./shaders/stamp.js";

const INSTANCE_FLOATS = 10;
const INSTANCE_BYTES = INSTANCE_FLOATS * Float32Array.BYTES_PER_ELEMENT;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Unable to create WebGL shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "Unknown WebGL shader error.";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, STAMP_VERTEX_SHADER);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, STAMP_FRAGMENT_SHADER);
  const program = gl.createProgram();
  if (!program) throw new Error("Unable to create WebGL program.");
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || "Unable to link WebGL stamp program.";
    gl.deleteProgram(program);
    throw new Error(message);
  }
  return program;
}

export function createBrushTexture(gl, image) {
  const texture = gl.createTexture();
  if (!texture) throw new Error("Unable to create brush texture.");
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.generateMipmap(gl.TEXTURE_2D);
  return texture;
}

export class StampRenderer {
  constructor(gl) {
    this.gl = gl;
    this.program = createProgram(gl);
    this.vertexArray = gl.createVertexArray();
    this.vertexBuffer = gl.createBuffer();
    this.instanceBuffer = gl.createBuffer();
    if (!this.vertexArray || !this.vertexBuffer || !this.instanceBuffer) {
      throw new Error("Unable to allocate WebGL stamp buffers.");
    }
    gl.bindVertexArray(this.vertexArray);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -0.5, -0.5, 0, 0,
       0.5, -0.5, 1, 0,
      -0.5,  0.5, 0, 1,
       0.5,  0.5, 1, 1,
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    for (const [location, size, offset] of [
      [2, 2, 0], [3, 2, 8], [4, 1, 16], [5, 4, 20], [6, 1, 36],
    ]) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, INSTANCE_BYTES, offset);
      gl.vertexAttribDivisor(location, 1);
    }
    gl.bindVertexArray(null);
    this.viewportLocation = gl.getUniformLocation(this.program, "u_viewport");
    this.brushLocation = gl.getUniformLocation(this.program, "u_brush");
  }

  drawLayer({ texture, stamps, width, height }) {
    if (!texture || !Array.isArray(stamps) || stamps.length === 0) return 0;
    const { gl } = this;
    const data = new Float32Array(stamps.length * INSTANCE_FLOATS);
    stamps.forEach((stamp, index) => {
      const offset = index * INSTANCE_FLOATS;
      const color = stamp.color || [1, 1, 1];
      data[offset] = stamp.x;
      data[offset + 1] = stamp.y;
      data[offset + 2] = stamp.width ?? stamp.size;
      data[offset + 3] = stamp.height ?? stamp.size;
      data[offset + 4] = stamp.rotation ?? 0;
      data[offset + 5] = color[0];
      data[offset + 6] = color[1];
      data[offset + 7] = color[2];
      data[offset + 8] = stamp.opacity ?? 1;
      data[offset + 9] = stamp.colorize === false ? 0 : 1;
    });

    gl.useProgram(this.program);
    gl.uniform2f(this.viewportLocation, width, height);
    gl.uniform1i(this.brushLocation, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindVertexArray(this.vertexArray);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, stamps.length);
    gl.bindVertexArray(null);
    return stamps.length;
  }

  dispose() {
    const { gl } = this;
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteBuffer(this.instanceBuffer);
    gl.deleteVertexArray(this.vertexArray);
    gl.deleteProgram(this.program);
  }
}
