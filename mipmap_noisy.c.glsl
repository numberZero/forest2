#version 450

layout(local_size_x = 4, local_size_y = 4) in;

layout(binding = 7, rgba16f) readonly restrict uniform image2DMS orig;
layout(binding = 0) writeonly restrict uniform image2D mips[5];

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash(uint x) {
	x += x << 10u;
	x ^= x >> 6u;
	x += x << 3u;
	x ^= x >> 11u;
	x += x << 15u;
	return x;
}

uint hash(uvec2 v) { return hash(v.x ^ hash(v.y)); }
uint hash(uvec3 v) { return hash(v.x ^ hash(v.y ^ hash(v.z))); }
uint hash(uvec4 v) { return hash(v.x ^ hash(v.y ^ hash(v.z ^ hash(v.w)))); }

float floatConstruct(uint m) {
	const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
	const uint ieeeOne = 0x3F800000u; // 1.0 in IEEE binary32

	m &= ieeeMantissa; // Keep only mantissa bits (fractional part)
	m |= ieeeOne; // Add fractional part to 1.0

	float  f = uintBitsToFloat(m); // Range [1:2]
	return f - 1.0; // Range [0:1]
}

vec4 imageLoadMS(readonly restrict image2DMS image, ivec2 coord) {
	uint n = imageSamples(image);
	int k = int(hash(uvec2(coord)) % n);
	return imageLoad(image, coord, k);
}

ivec2 pos[5];
vec4 accums[6][4];
int accums_usage[6];
vec4 current;

void accum_begin(int level) {
	accums_usage[level] = 0;
	for (int k = 0; k < 4; k++)
		accums[level][k] = vec4(0.0);
}

int random(int level) {
	return int(hash(uvec3(pos[level], level)) % 4u);
}

int wrandom(int level) {
	float ws = 0.0;
	for (int k = 0; k < 4; k++)
		ws += accums[level][k].w;
	float w = ws * floatConstruct(hash(uvec3(pos[level], level)));
	for (int k = 0; k < 4; k++) {
		w -= accums[level][k].w;
		if (w <= 0.0)
			return k;
	}
	return 3;
}

void accum_end(int level) {
	current = accums[level][wrandom(level)];
}

void accum(int level) {
	accums[level][accums_usage[level]++] = current;
}

void load(int level, vec4 value) {
	current = value;
}

void store(int level) {
	imageStore(mips[level], pos[level], current);
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

	texels[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = current;
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
