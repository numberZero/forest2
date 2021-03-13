#version 450

layout(local_size_x = 4, local_size_y = 4) in;

layout(binding = 7, rgba16f) readonly restrict uniform image2DMS orig;
layout(binding = 0) writeonly restrict uniform image2D mips[5];

layout(location = 0) uniform bool normal = false;

vec4 imageLoadMS(readonly restrict image2DMS image, ivec2 coord) {
	int n = imageSamples(image);
	vec4 result = vec4(0.0);
	for (int k = 0; k < n; k++)
		result += imageLoad(image, coord, k);
	return result / n;
}

ivec2 pos[5];
vec4 accums[6];

void accum_begin(int level) {
	accums[level] = vec4(0.0);
}

void accum_end(int level) {
	accums[level] /= 4.0;
}

void accum(int level) {
	vec4 prev = accums[level - 1];
	if (normal)
		prev.xyz = normalize(prev.xyz) * prev.w;
	accums[level] += prev;
}

void load(int level, vec4 value) {
	accums[level] = value;
}

void store(int level) {
	vec4 value = accums[level];
	if (normal)
		value.xyz = normalize(value.xyz);
	imageStore(mips[level], pos[level], value);
}

const ivec2 offs[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
shared vec4 texels[4][4];

void main() {
	pos[4] = ivec2(gl_WorkGroupID.xy);

	pos[2] = 4 * pos[4] + ivec2(gl_LocalInvocationID.xy);
	accum_begin(2);
	for (int l = 0; l < 4; l++) {
		pos[1] = 2 * pos[2] + offs[l];
		accum_begin(1);
		for (int k = 0; k < 4; k++) {
			pos[0] = 2 * pos[1] + offs[k];
			load(0, imageLoadMS(orig, pos[0] + offs[k]));
			store(0);
			accum(1);
		}
		accum_end(1);
		store(1);
		accum(2);
	}
	accum_end(2);
	store(2);

	texels[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = accums[2];
	barrier();
	if (gl_LocalInvocationIndex != 0)
		return;

	accum_begin(4);
	for (int l = 0; l < 4; l++) {
		pos[3] = 2 * pos[4] + offs[l];
		accum_begin(3);
		for (int k = 0; k < 4; k++) {
			pos[2] = 2 * pos[2] + offs[k];
			ivec2 off = 2 * offs[l] + offs[k];
			load(2, texels[off.x][off.y]);
			accum(3);
		}
		accum_end(3);
		store(3);
		accum(4);
	}
	accum_end(4);
	store(4);
}
