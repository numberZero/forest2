#version 460

layout(binding = 0) uniform sampler2D colors;
layout(binding = 1) uniform sampler2D transparencies;

out vec4 o_color;

void main() {
	vec4 color = texelFetch(colors, ivec2(gl_FragCoord.xy), 0);
	float transparency = texelFetch(transparencies, ivec2(gl_FragCoord.xy), 0).x;
	if (color.a == 0.0)
		discard;
	o_color = vec4(1.0 * color.rgb / color.a, 1.0 - transparency);
// 	o_color = vec4(color.a, 10.0 * color.g, 1.0 - transparency, 1.0);
}
