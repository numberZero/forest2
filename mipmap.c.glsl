#version 450

layout(local_size_x = 4, local_size_y = 4) in;

layout(binding = 7, rgba16f) readonly restrict uniform image2DMS orig;
layout(binding = 0) writeonly restrict uniform image2D level[5];

layout(location = 0) uniform bool normal = false;

vec4 imageLoadMS(readonly restrict image2DMS image, ivec2 coord) {
	int n = imageSamples(image);
	vec4 result = vec4(0.0);
	for (int k = 0; k < n; k++)
		result += imageLoad(image, coord, k);
	return result / n;
}

void subscale(inout vec4 sum) {
	if (normal)
		sum.xyz = normalize(sum.xyz);
	else
		sum.xyz /= 4.0;
	sum.w /= 4.0;
}

const ivec2 offs[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
shared vec4 texels[4][4];

void main() {
	ivec2 level4_pos = ivec2(gl_WorkGroupID.xy);
	vec4 accums[7];

	ivec2 level2_pos = 4 * level4_pos + ivec2(gl_LocalInvocationID.xy);
	accums[2] = vec4(0.0);
	for (int l = 0; l < 4; l++) {
		ivec2 level1_pos = 2 * level2_pos + offs[l];
		accums[1] = vec4(0.0);
		for (int k = 0; k < 4; k++) {
			ivec2 level0_pos = 2 * level1_pos + offs[k];
			accums[0] = imageLoadMS(orig, level0_pos + offs[k]);
			imageStore(level[0], level0_pos, accums[0]);
			accums[1] += accums[0];
		}
		subscale(accums[1]);
		imageStore(level[1], level1_pos, accums[1]);
		accums[2] += accums[1];
	}
	subscale(accums[2]);
	imageStore(level[2], level2_pos, accums[2]);

	texels[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = accums[2];
	barrier();
	if (gl_LocalInvocationIndex != 0)
		return;

	accums[4] = vec4(0.0);
	for (int l = 0; l < 4; l++) {
		ivec2 level3_pos = 2 * level4_pos + offs[l];
		accums[1] = vec4(0.0);
		for (int k = 0; k < 4; k++) {
			ivec2 level2_pos = 2 * level2_pos + offs[k];
			ivec2 off = 2 * offs[l] + offs[k];
			accums[2] = texels[off.x][off.y];
			accums[3] += accums[2];
		}
		subscale(accums[3]);
		imageStore(level[3], level3_pos, accums[3]);
		accums[4] += accums[3];
	}
	subscale(accums[4]);
	imageStore(level[4], level4_pos, accums[4]);
}
