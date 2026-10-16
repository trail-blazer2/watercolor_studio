import { createWebGL2Context } from "../gpu/context.js";
import { StampRenderer, createBrushTexture } from "../gpu/stamp.js";

const vertex = `#version 300 es
precision highp float;
out vec2 uv;
void main() {
  vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
  uv=p; gl_Position=vec4(p*2.-1.,0.,1.);
}`;
const fragment = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec3 correction;
in vec2 uv;
out vec4 color;
void main() {
  vec4 c=texture(image,uv);
  vec3 rgb=c.a>0. ? c.rgb/c.a : vec3(0.);
  float gray=dot(rgb,vec3(.299,.587,.114));
  rgb=mix(vec3(gray),rgb,correction.z);
  rgb=clamp((rgb*correction.x-.5)*correction.y+.5,0.,1.);
  color=vec4(rgb*c.a,c.a);
}`;

function program(gl) {
  const p = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s); gl.deleteShader(s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

export function makeCanvas(width = 1, height = 1) {
  const canvas = typeof OffscreenCanvas === "function" ? new OffscreenCanvas(width, height) :
    globalThis.document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  return canvas;
}

export class PaintingCompositor {
  constructor(assets) {
    this.canvas = makeCanvas();
    this.gl = createWebGL2Context(this.canvas, { preserveDrawingBuffer: true }).gl;
    this.stamps = new StampRenderer(this.gl);
    this.program = program(this.gl);
    this.vao = this.gl.createVertexArray();
    this.textures = new Map();
    this.assetBytes = 0;
    for (const [name, image] of assets) {
      this.textures.set(name, createBrushTexture(this.gl, image));
      this.assetBytes += image.width * image.height * 4 * 4 / 3;
    }
    const gl = this.gl;
    this.input = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.input);
    for (const [parameter, value] of [
      [gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
    ]) gl.texParameteri(gl.TEXTURE_2D, parameter, value);
    this.imageUniform = gl.getUniformLocation(this.program, "image");
    this.correctionUniform = gl.getUniformLocation(this.program, "correction");
  }
  resize(width, height) {
    const max = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);
    // Reuse two CPU surfaces and two GPU buffers; never cache full-resolution layers.
    if (width > max || height > max || width * height * 24 + this.assetBytes > 250 * 1024 * 1024) {
      throw new RangeError("Render exceeds the painting memory budget; use tiled export.");
    }
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
  }
  drawStamps(name, stamps, width, height) {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    this.stamps.drawLayer({ texture: this.textures.get(name), stamps, width, height });
    return this.canvas;
  }
  correct(canvas, params) {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.useProgram(this.program); gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.input);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.uniform1i(this.imageUniform, 0);
    gl.uniform3f(this.correctionUniform, params.brightness ?? 1, params.contrast ?? 1, params.saturation ?? 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
    return this.canvas;
  }
  dispose() {
    const gl = this.gl;
    this.stamps.dispose();
    for (const texture of this.textures.values()) gl.deleteTexture(texture);
    gl.deleteTexture(this.input); gl.deleteProgram(this.program); gl.deleteVertexArray(this.vao);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    this.canvas.width = this.canvas.height = 1;
  }
}
