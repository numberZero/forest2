#version 460

layout(location = 0) uniform mat4 projection_matrix;
layout(location = 1) uniform mat4 camera_matrix;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;

out vec4 color;

void main() {
	color = vec4(in_color, 1.0);
	gl_Position = projection_matrix * camera_matrix * vec4(in_pos, 1.0);
}
