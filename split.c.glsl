#version 450

layout(local_size_x = 1) in;

layout(location = 0) uniform vec3 camera_position;
layout(location = 1) uniform float threshold = 15.0;

layout(binding = 0) readonly restrict buffer SourcePositions {
	vec4 position[];
};

layout(binding = 1) uniform atomic_counter near_objects;
layout(binding = 1) writeonly restrict buffer NearPositions {
	vec4 npos[];
};

layout(binding = 2) uniform atomic_counter far_objects;
layout(binding = 2) writeonly restrict buffer FarPositions {
	vec4 fpos[];
};

void main() {
	vec4 pos = position[gl_GlobalInvocationID.x];
	float dist = distance(camera_position, pos.xyz);
	if (dist < threshold) {
		uint index = atomicCounterIncrement(near_objects);
		npos[index] = pos;
	} else {
		uint index = atomicCounterIncrement(far_objects);
		fpos[index] = pos;
	}
}
