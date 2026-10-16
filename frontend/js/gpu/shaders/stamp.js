export const STAMP_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec2 a_center;
layout(location = 3) in vec2 a_size;
layout(location = 4) in float a_rotation;
layout(location = 5) in vec4 a_tint;
layout(location = 6) in float a_colorize;
uniform vec2 u_viewport;
out vec2 v_uv;
out vec4 v_tint;
out float v_colorize;
void main() {
  float c = cos(a_rotation);
  float s = sin(a_rotation);
  vec2 local = a_position * a_size;
  vec2 rotated = vec2(c * local.x + s * local.y, -s * local.x + c * local.y);
  vec2 pixel = a_center + rotated;
  vec2 clip = vec2(pixel.x * 2.0 / u_viewport.x - 1.0, 1.0 - pixel.y * 2.0 / u_viewport.y);
  gl_Position = vec4(clip, 0.0, 1.0);
  v_uv = a_uv;
  v_tint = a_tint;
  v_colorize = a_colorize;
}`;

export const STAMP_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform sampler2D u_brush;
in vec2 v_uv;
in vec4 v_tint;
in float v_colorize;
out vec4 outColor;
void main() {
  vec4 brush = texture(u_brush, v_uv);
  float alpha = brush.a * v_tint.a;
  vec3 premultiplied = mix(brush.rgb * v_tint.a,
    v_tint.rgb * brush.a * v_tint.a, step(0.5, v_colorize));
  outColor = vec4(premultiplied, alpha);
}`;
