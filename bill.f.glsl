#version 460

layout(location = 4) uniform float threshold = 0.125;
layout(binding = 0) uniform sampler2DArray tex;

in vec2 uv;
in flat int layer;
in vec4 color;
out vec4 o_color;

void main() {
	vec4 c = texture(tex, vec3(uv, layer)) * color;
	if (c.a <= threshold)
		discard;
	o_color = c;
}
