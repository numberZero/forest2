#version 460

layout(binding = 0) uniform sampler2DArray tex;

in vec2 uv;
in flat int layer;
in vec4 color;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_transparency;

void main() {
	vec4 c = texture(tex, vec3(uv, layer)) * color;
	if (c.a <= 0.125)
		discard;
	float w = exp((gl_FragCoord.w - 1.0) * 2.0);
	o_color = c * w;
	o_transparency = 1.0 - c.a;
}
