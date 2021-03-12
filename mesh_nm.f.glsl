#version 460

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal;

in vec3 f_color;
in vec3 w_normal;

void main() {
	out_color = vec4(f_color, 1.0);
	out_normal = vec4(w_normal, 1.0);
}
