#version 460

#define PI 3.1415926535897932384626433832795

layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

layout(location = 0) uniform mat4 projection_matrix;
layout(location = 1) uniform mat3 camera_matrix;
layout(location = 2) uniform vec3 camera_position;
layout(location = 3) uniform int angles;

in PerVertex
{
	in vec3 position;
	in vec2 size;
} vert[];

out vec2 uv1, uv2;
out flat int layer1, layer2;
out flat float delta;

vec3 position;
vec2 size;

const ivec2 verts[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

void main() {
	const float single_view_angle = PI / angles;

	position = vert[0].position;
	size = vert[0].size;
	vec3 rel_position = position - camera_position;
	vec3 visual_position = camera_matrix * rel_position;
	float v_angle = atan(length(rel_position.xy), -1.0-rel_position.z);
	float off0 = sin(v_angle);

	int view = clamp(int(round(v_angle / PI * angles)), 0, angles / 2);
	float angle = float(view) * PI / angles;
	float off1 = sin(angle);

// 	vec2 camera_direction = normalize(transpose(camera_matrix)[1].xy);
	vec2 dir = normalize(rel_position.xy);
	vec3 u = vec3(-dir.y, dir.x, 0.0);
	vec3 v = vec3(cos(angle) * dir, sin(angle));

	for (int vid = 0; vid < 4; vid++) {
		vec2 vertex = vec2(verts[vid]);
		vec2 p = vertex * 2.0 - 1.0;
		p.y += 0.5 * off1;
// 		vec3 pos = transpose(camera_matrix) * vec3(p.x, 0.0, p.y);
		vec3 pos = u * p.x + v * p.y;
		vec4 spos = vec4(camera_matrix * (rel_position + pos), 1.0);
		gl_Position = projection_matrix * spos;
		uv1 = uv2 = vertex;
		layer1 = layer2 = view;
		delta = 0.0;
// 		mark = 0.5 - 0.25 * off1;
		EmitVertex();
	}
/*

	float view_coord = v_angle / single_view_angle;
	int view1 = int(floor(view_coord));
	int view2 = view1 + 1;
	float view_frac = view_coord - view1;
	float off1 = sin(view1 * single_view_angle);
	float off2 = sin(view2 * single_view_angle);
	if (off1 < off2) {
		float o = off1;
		off1 = off2;
		off2 = o;
		view1++;
		view2--;
		view_frac = 1.0 - view_frac;
	}

	float uvoff = 0.5 * (off1 - off2);
	for (int vid = 0; vid < 4; vid++) {
		vec2 vertex = vec2(verts[vid]);
		vec3 pos = vec3(vertex * 2.0 - 1.0, 0.0);
		if (vid < 2)
			pos.y += 0.5 * off2;
		else
			pos.y += 0.5 * off1;
		pos.z -= 0.5 * off0;
		gl_Position = projection_matrix * vec4(pos.xzy + visual_position, 1.0);
		uv1 = vertex;
		uv2 = vertex;
		if (vid < 2)
			uv1.y -= 0.5 * uvoff;
		else
			uv2.y += 0.5 * uvoff;
		layer1 = view1;
		layer2 = view2;
		delta = view_frac;
		EmitVertex();
	}*/
}
