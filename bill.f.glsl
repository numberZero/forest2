#version 460

layout(location = 4) uniform float threshold = 0.125;
layout(location = 5) uniform int mode = 1;
layout(location = 6) uniform vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
layout(location = 7) uniform vec3 ambi_color = vec3(0.5, 0.5, 0.5);
layout(location = 8) uniform vec3 light_color = vec3(0.5, 0.5, 0.5);
layout(binding = 0) uniform sampler2DArray tex;
layout(binding = 1) uniform sampler2DArray nm;

in vec2 uv;
in flat int layer;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_transparency;

void main() {
	vec4 c = texture(tex, vec3(uv, layer));
	if (c.a <= threshold)
		discard;
	vec4 n = texture(nm, vec3(uv, layer));
	vec3 normal = normalize(n.xyz);
	if (any(isnan(normal)))
		normal = vec3(0.0);
	float light = max(0.0, dot(normal, light_dir));
	float ambilight = 0.8 + 0.2 * normal.z;
	c.rgb *= ambilight * ambi_color + light * light_color;
	if (mode == 0)
		c.rgb /= c.a;
	if (mode == 2) {
		float dist = gl_FragCoord.z / gl_FragCoord.w;
		o_transparency = 1.0 - c.a;
		c *= exp(10.0 - 0.1 * dist);
	}
	o_color = c;
}
