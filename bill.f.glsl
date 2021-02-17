#version 460

layout(binding = 0) uniform sampler2DArray tex;

in vec2 uv;
in flat int layer;
in float delta;
out vec4 o_color;

void main() {
	vec4 c = texture(tex, vec3(uv, layer));
	if (c.a <= 0.125)
		discard;
	c.a = min(c.a * 2, 1.0);
	o_color = c;
}
