#version 460

layout(location = 0) uniform mat4 projection_matrix;
layout(location = 1) uniform mat3 camera_matrix;
layout(location = 2) uniform vec3 camera_position;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 size;

out PerVertex
{
	vec3 position;
	vec2 size;
} vert;

void main() {
	vert.position = position;
	vert.size = size;
}
