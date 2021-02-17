#!/usr/bin/env python3

from sys import argv
import os.path
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
		itype = 'uint16'
		if nverts > 65536:
			print('Oops, more than 64ki vertices in an object. Unsupported.')
			return # FIXME
			itype = 'uint32'
		indices = np.array(self.indices, dtype=itype)
		vmap = np.ndarray((nverts, 3), dtype='int')
		for i, v in enumerate(self.vmap.keys()):
			vmap[i] = v
		vmap -= 1
		vertices = np.concatenate([np.array(self.a[a], dtype='float32')[vmap[:, a]] for a in range(3)], axis=1)
		assert(vertices.shape == (nverts, 8))
		self.objects.append({
			'name': self.oname,
			'vertices': vertices,
			'indices': indices,
			})

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
	name, _ = os.path.splitext(os.path.basename(filename))
	f = ObjFile()
	f.load(lines)
	for k, obj in enumerate(f.objects):
		for aname in 'vertices', 'indices':
			obj[aname].tofile(f'{name}.{k}.{aname[0]}')
