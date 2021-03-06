#version 460

layout(location = 0) uniform mat4 vp_matrix;
layout(location = 1) uniform mat4 m_matrix = mat4(1.0);
layout(location = 2) uniform vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_offset;
layout(location = 3) in vec3 in_normal;

out vec4 color;

void main() {
	color = vec4(in_color * (0.5 + 0.5 * dot(in_normal, light_dir)), 1.0);
	gl_Position = vp_matrix * (vec4(in_offset, 0.0) + m_matrix * vec4(in_pos, 1.0));
}
