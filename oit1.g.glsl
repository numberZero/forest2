#version 460

layout(points) in;
layout(triangle_strip, max_vertices = 6) out;

// out vec2 uv;

void main() {
// 	uv = vec2(1.0, 0.0);
	gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
	EmitVertex();

// 	uv = vec2(1.0, 1.0);
	gl_Position = vec4(1.0, 1.0, 0.0, 1.0);
	EmitVertex();

// 	uv = vec2(0.0, 0.0);
	gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
	EmitVertex();

// 	uv = vec2(0.0, 1.0);
	gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);
	EmitVertex();
}
