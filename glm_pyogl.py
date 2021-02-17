import ctypes
import glm
from OpenGL.GL import *
from OpenGL.arrays.formathandler import FormatHandler

class VectorHandler(FormatHandler):
	dataPointer = staticmethod(glm.value_ptr)

	def __init__(self, typ, l):
		self.glm_type = typ
		self.ctype = ctypes.c_float * l
		self.l = l

	def arrayToGLType(self, array, typeCode=None):
		return GL_FLOAT

	def arraySize(self, array, typeCode=None):
		return self.l

	def arrayByteCount(self, array, typeCode=None):
		return self.l * 4

	def asArray(self, value, typeCode=None):
		return self.ctype(*value.to_tuple())

	def unitSize(self, value, typeCode=None):
		return self.l

class MatrixHandler(FormatHandler):
	dataPointer = staticmethod(glm.value_ptr)

	def __init__(self, typ, w, h):
		self.glm_type = typ
		self.ctype = ctypes.c_float * (w * h)
		self.w = w
		self.h = h
		self.l = w * h

	def arrayToGLType(self, array, typeCode=None):
		return GL_FLOAT

	def arraySize(self, array, typeCode=None):
		return self.l

	def arrayByteCount(self, array, typeCode=None):
		return self.l * 4

	def asArray(self, value, typeCode=None):
		return self.ctype(*sum(value.to_tuple(), ()))

	def unitSize(self, value, typeCode=None):
		return self.l

for l in range(2, 5):
	typ = getattr(glm, f'vec{l}')
	VectorHandler(typ, l).register((typ,))

for w in range(2, 5):
	for h in range(2, 5):
		typ = getattr(glm, f'mat{w}x{h}')
		MatrixHandler(typ, w, h).register((typ,))
