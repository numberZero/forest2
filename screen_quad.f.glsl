#version 420

layout(binding = 0) uniform sampler2D colors;

out vec4 o_color;

void main() {
	o_color = texelFetch(colors, ivec2(gl_FragCoord.xy), 0);
}
