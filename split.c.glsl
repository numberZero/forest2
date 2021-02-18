#version 450

layout(std430) struct DrawElementsIndirectCommand {
	uint count;
	uint instanceCount;
	uint firstIndex;
	uint baseVertex;
	uint baseInstance;
};

layout(std430) struct DrawArraysIndirectCommand {
	uint count;
	uint instanceCount;
	uint first;
	uint baseInstance;
};

layout(std430) struct Indirect {
	DrawElementsIndirectCommand near;
	DrawArraysIndirectCommand far;
};

layout(local_size_x = 1) in;

layout(location = 0) uniform vec3 camera_position;
layout(location = 1) uniform float threshold = 15.0;

layout(binding = 0) readonly restrict buffer SourcePositions {
	vec4 position[];
};

layout(binding = 1) writeonly restrict buffer NearPositions {
	vec4 npos[];
};

layout(binding = 2) writeonly restrict buffer FarPositions {
	vec4 fpos[];
};

layout(binding = 3) restrict buffer Indirects {
	Indirect inds[];
};

void main() {
	vec4 pk = position[gl_GlobalInvocationID.x];
	vec4 pos = vec4(pk.xyz, 1.0);
	uint kind = uint(pk.w);
	float dist = distance(camera_position, pos.xyz);
	if (dist < threshold) {
		uint index = atomicAdd(inds[kind].near.instanceCount, 1);
		npos[inds[kind].near.baseInstance + index] = pos;
	} else {
		uint index = atomicAdd(inds[kind].far.count, 1);
		fpos[inds[kind].far.first + index] = pos;
	}
}
