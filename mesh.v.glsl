#version 460

layout(location = 0) uniform mat4 vp_matrix;
layout(location = 1) uniform mat4 m_matrix = mat4(1.0);
layout(location = 2) uniform vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
layout(location = 3) uniform vec3 ambi_color = vec3(0.5, 0.5, 0.5);
layout(location = 4) uniform vec3 light_color = vec3(0.5, 0.5, 0.5);

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_offset;
layout(location = 3) in vec3 in_normal;

out vec4 color;
out vec3 f_color;
out vec3 w_normal;

void main() {
	f_color = in_color;
	w_normal = vec3(m_matrix * vec4(in_normal, 0.0));
	float light = max(0.0, dot(w_normal, light_dir));
	color = vec4(in_color * (ambi_color + light * light_color), 1.0);
	gl_Position = vp_matrix * (vec4(in_offset, 0.0) + m_matrix * vec4(in_pos, 1.0));
}
