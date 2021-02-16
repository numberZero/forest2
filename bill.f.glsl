#version 460

layout(binding = 0) uniform sampler2DArray tex;

in vec2 uv1, uv2;
in flat int layer1, layer2;
in float delta;
out vec4 o_color;

void main() {
	vec4 c1 = texture(tex, vec3(uv1, layer1));
	vec4 c2 = texture(tex, vec3(uv2, layer2));
	vec4 c = mix(c1, c2, delta);
	if (c.a <= 0.125)
		discard;
	c.a = min(c.a * 2, 1.0);
	o_color = c;
}
