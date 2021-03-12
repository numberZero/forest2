#version 460

layout(location = 4) uniform float threshold = 0.125;
layout(location = 5) uniform int mode = 1;
layout(binding = 0) uniform sampler2DArray tex;

in vec2 uv;
in flat int layer;
in vec4 color;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_transparency;

void main() {
	vec4 c = texture(tex, vec3(uv, layer)) * color;
	if (c.a <= threshold)
		discard;
	if (mode == 0)
		c.rgb /= c.a;
	if (mode == 2) {
		float dist = gl_FragCoord.z / gl_FragCoord.w;
		o_transparency = 1.0 - c.a;
		c *= exp(10.0 - 0.1 * dist);
	}
	o_color = c;
}
