#!/usr/bin/env python3

from sys import argv
import re
import glm
import numpy as np

class ObjFile:
	def _cmd_o(self, name):
		self._fini_object()
		self._init_object(name)

	def _cmd_v(self, x, y, z = 0.0):
		self.v.append(glm.vec3(float(x), float(y), float(z)))

	def _cmd_vt(self, x, y):
		self.vt.append(glm.vec2(float(x), float(y)))

	def _cmd_vn(self, x, y, z):
		self.vn.append(glm.vec3(float(x), float(y), float(z)))

	def _cmd_f(self, *verts):
		index0 = self._vertex(verts[0])
		index1 = self._vertex(verts[1])
		for vdesc in verts[2:]:
			index = self._vertex(vdesc)
			self.indices.extend([index0, index1, index])
			index1 = index

	def _vertex(self, desc):
		v, vt, vn = tuple(map(int, desc.split('/')))
		key = v, vt, vn
		index = self.vmap.get(key)
		if not index:
			index = len(self.vmap)
			self.vmap[key] = index
		return index

	def _init_object(self, name):
		self.oname = name
		self.f = []
		self.vmap = {}
		self.indices = []

	def _fini_object(self):
		if not self.vmap:
			return
		nverts = len(self.vmap)
		ninds = len(self.indices)
		print(f'{nverts} vertices, {ninds} indices in object {self.oname}')
		if nverts > 65536:
			print('Oops, more than 64ki vertices in an object. That\'s not supported.')
		indices = np.array(self.indices, dtype='uint16')
		vmap = np.ndarray((nverts, 3), dtype='int')
		for i, v in enumerate(self.vmap.keys()):
			vmap[i] = v
		vmap -= 1
		vertices = np.concatenate([np.array(self.a[a])[vmap[:, a]] for a in range(3)], axis=1)
		assert(vertices.shape == (nverts, 8))

	def __init__(self):
		self.v = []
		self.vt = []
		self.vn = []
		self.a = (self.v, self.vt, self.vn)
		self.objects = []

	def load(self, lines: [str]):
		self._init_object('default')
		for line in lines:
			line = line.strip()
			if not line or line[0] == '#':
				continue
			parts = line.split()
			command = parts[0]
			args = parts[1:]
			try:
				getattr(self, f'_cmd_{command}')(*args)
			except AttributeError as e:
				print(e)
				#print(f'Warning: command not found: {command}')
		self._fini_object()

for filename in argv[1:]:
	print(f'Importing {filename}')
	with open(filename, 'r') as f:
		lines = f.readlines()
	f = ObjFile()
	f.load(lines)

{
	'mtllib': ['maple.mtl'],
	'o': ['leaves.001_leaves.004'],
	'v': ['0.046464', '0.192974', '1.853366'],
	'vt': ['0.600000', '0.666667'],
	'vn': ['0.4806', '0.7007', '-0.5272'],
	'usemtl': ['Material.001'],
	's': ['off'],
	'f': ['73267/75891/26923', '73272/75895/26923', '73271/75896/26923', '73270/75892/26923']
}
