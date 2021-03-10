#version 460

#define PI 3.1415926535897932384626433832795

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) uniform mat4 projection_matrix;
layout(location = 1) uniform mat3 camera_matrix;
layout(location = 2) uniform vec3 camera_position;
layout(location = 3) uniform int v_view_per_halfcircle;

layout(binding = 0, std430) readonly restrict buffer h_view_counts_buf {
	int h_view_ends[];
};

layout(binding = 1, std430) readonly restrict buffer opm_buf {
	mat4 orig_proj_matrices[];
};

in PerVertex
{
	in vec3 position;
	in vec2 size;
} vert[];

out vec2 uv;
out vec4 color;
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
	vec4 c = {1.0, 1.0, 1.0, 1.0};

	int v_view_max = h_view_ends.length() - 1;
	float v_view_step = PI / v_view_per_halfcircle;
	float v_view_angle = atan(length(rel_position.xy), -rel_position.z - 1.0); // 0 — сверху, π — снизу
	int v_view = clamp(int(round(v_view_angle / v_view_step)), 0, v_view_max); // [0 .. v_view_max]
	float v_angle = v_view * v_view_step;

	float h_view_angle;

	if (v_view != 0) { // v_view ∈ [1 .. v_view_max]
		int h_view_base = h_view_ends[v_view - 1];
		int h_view_count = h_view_ends[v_view] - h_view_base;

		float h_view_step = 2.0 * PI / h_view_count;
		h_view_angle = atan(rel_position.x, rel_position.y);
		int h_view = int(round(h_view_angle / h_view_step));
		float h_angle = h_view * h_view_step;
		if (h_view < 0)
			h_view += h_view_count;
		h_view %= h_view_count; // на отрицательных числах выдаёт ересь

		view = h_view_base + h_view;
	}

	vec3 world_offset_base_center = vec3(0.0, 0.0, 0.5);

	vec4 cpos = vec4(camera_matrix * (rel_position + world_offset_base_center), 1.0);
	vec4 cpos_view = projection_matrix * cpos;

	mat4 view_matrix = mat4(camera_matrix);
	view_matrix[3].xyz = -camera_matrix * camera_position;

	vec3 camera_direction = transpose(camera_matrix)[1];
	vec3 direction = normalize(rel_position);
	if (dot(direction, camera_direction) < 0.5)
		return;

	mat4 orig_proj_matrix = orig_proj_matrices[view];

	mat3 w;

	const float v_threshold = 0.5;
	if (v_view_angle < v_threshold) {
		mat4x3 model_matrix = mat4x3(size.x);
		model_matrix[2].z = size.y;
		model_matrix[3].xyz = position;

		const float r = 0.5;
		const float h = 0.5;

		/* Полная сфера
		const float S = 4.0 * PI * r * r;
		mat4 I = mat4(r*r/3.0 * S);
		I[2].z += h*h * S;
		I[3].z = I[2].w = h * S;
		I[3].w = S;
		*/

		/* Полусфера
		*/
		const float S = 2.0 * PI * r * r;
		mat4 I = mat4(r*r/3.0);
		I[2].z += r*h + h*h;
		I[3].z = I[2].w = r/2.0 + h;
		I[3].w = 1.0;
		I *= S;

		/* Выбор текстуры
		Оптимальный до перспективных искажений — одинаковый для всех точек экрана...
		float E0 = 1.0e5;
		for (int k = 0; k < orig_proj_matrices.length(); k++) {
			mat4 opm = orig_proj_matrices[k];
			mat4x3 R = mat4x3(opm[0].xzw, opm[1].xzw, opm[2].xzw, opm[3].xzw);
			mat3x4 m = I * transpose(R) * inverse(R * I * transpose(R));
			mat4 D = view_matrix * mat4(model_matrix) * (mat4(m * R) - mat4(1.0));
			mat4 EE = D * I * transpose(D);
			float E = EE[0].x + EE[2].z;
			if (E < E0) {
				E0 = E;
				view = k;
			}
		}
		orig_proj_matrix = orig_proj_matrices[view];
		*/

		mat4x3 R = mat4x3(orig_proj_matrix[0].xzw, orig_proj_matrix[1].xzw, orig_proj_matrix[2].xzw, orig_proj_matrix[3].xzw);
		mat3x4 m = I * transpose(R) * inverse(R * I * transpose(R));
		w = model_matrix * m;
	} else {
		vec2 farther_horizontally = normalize(rel_position.xy);
		vec3 key_points_model[3] = {
			vec3(0.0, 0.0, 0.0),
			vec3(0.0, 0.0, 1.0),
			vec3(0.5 * farther_horizontally.y, -0.5 * farther_horizontally.x, 0.5),
		};

		vec3 key_points_world[3];
		vec2 key_points_projected[3];
		for (int k = 0; k < 3; k++) {
			vec3 kp_scaled = key_points_model[k];
			kp_scaled.xy *= size.x;
			kp_scaled.z *= size.y;
			key_points_world[k] = position + kp_scaled;
			key_points_projected[k] = (orig_proj_matrix * vec4(key_points_model[k], 1.0)).xz;
		}
		mat3 m1, m2;
		for (int k = 0; k < 3; k++) {
			m1[k] = key_points_world[k];
			m2[k] = vec3(key_points_projected[k], 1.0);
		}
		w = m1 * inverse(m2);
	}

	for (int vid = 0; vid < 4; vid++) {
		vec2 vertex = vec2(verts[vid]);
		vec3 pos = vec3(vertex * 2.0 - 1.0, 1.0);
		vec4 spos = vec4(camera_matrix * (w * pos - camera_position), 1.0);

		gl_Position = projection_matrix * spos;
		gl_Position.z = cpos_view.z * gl_Position.w / cpos_view.w;

		color = c;
		uv = vertex;
		layer = view;
		EmitVertex();
	}
}
