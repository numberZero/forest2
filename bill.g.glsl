#version 460

#define PI 3.1415926535897932384626433832795

layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

layout(location = 0) uniform mat4 projection_matrix;
layout(location = 1) uniform mat3 camera_matrix;
layout(location = 2) uniform vec3 camera_position;
layout(location = 3) uniform int v_view_per_halfcircle;

layout(binding = 0, std430) readonly restrict buffer h_view_counts_buf {
	int h_view_ends[];
};

in PerVertex
{
	in vec3 position;
	in vec2 size;
} vert[];

out vec2 uv;
out float delta;
out flat int layer;

vec3 position;
vec2 size;

const ivec2 verts[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

void main() {
	position = vert[0].position;
	size = vert[0].size;
	vec3 rel_position = position - camera_position;
	vec3 visual_position = camera_matrix * rel_position;

	int view = 0;
	vec3 u, v;

	int v_view_max = h_view_ends.length() - 1;
	float v_view_step = PI / v_view_per_halfcircle;
	float v_view_angle = atan(length(rel_position.xy), -rel_position.z - 1.0); // 0 — сверху, π — снизу
	int v_view = clamp(int(round(v_view_angle / v_view_step)), 0, v_view_max); // [0 .. v_view_max]
	float v_angle = v_view * v_view_step;

	if (v_view == 0) { // прямо сверху
		u = vec3(1.0, 0.0, 0.0);
		v = vec3(0.0, 1.0, 0.0);
	} else { // v_view ∈ [1 .. v_view_max]
		int h_view_base = h_view_ends[v_view - 1];
		int h_view_count = h_view_ends[v_view] - h_view_base;

		float h_view_step = 2.0 * PI / h_view_count;
		float h_view_angle = atan(rel_position.x, rel_position.y);
		int h_view = int(round(h_view_angle / h_view_step));
		float h_angle = h_view * h_view_step;
		if (h_view < 0)
			h_view += h_view_count;
		h_view %= h_view_count; // на отрицательных числах выдаёт ересь

		view = h_view_base + h_view;
// 		h_angle = h_view_angle;

		vec2 dir = vec2(sin(h_angle), cos(h_angle));
		u = vec3(dir.y, -dir.x, 0.0);
		v = vec3(cos(v_angle) * dir, sin(v_angle));
	}

	vec3 world_offset_base_center = vec3(0.0, 0.0, 0.5);
	float texture_offset_base_center = 0.5 * sin(v_angle);

	vec4 cpos = vec4(camera_matrix * (rel_position + world_offset_base_center), 1.0);
	vec4 cpos_view = projection_matrix * cpos;

	for (int vid = 0; vid < 4; vid++) {
		vec2 vertex = vec2(verts[vid]);
		vec2 p = vertex * 2.0 - 1.0;
		p.y += texture_offset_base_center;
		vec3 pos = u * p.x + v * p.y;
		vec4 spos = vec4(camera_matrix * (rel_position + pos), 1.0);
		gl_Position = projection_matrix * spos;
		gl_Position.z = cpos_view.z * gl_Position.w / cpos_view.w;

		delta = clamp(v_view_angle, 0.0, 1.0);
		uv = vertex;
		layer = view;
		EmitVertex();
	}
}
