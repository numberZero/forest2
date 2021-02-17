#!/usr/bin/python3

import os, os.path
import math
import ctypes
import glm
import glfw
import random
import numpy as np
import numpy.random as rnd
from OpenGL.GL import *
from OpenGL.error import Error
from OpenGL.arrays.formathandler import FormatHandler
from glm import vec2, vec3, vec4, mat2, mat3, mat4

TRULY_ISOMETRIC = False
FLY_CONTROLS = False
angles = 32

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

def make_camera_orientation_matrix(yaw: float, pitch: float = 0.0, roll: float = 0.0):
	from math import sin, cos
	yaw = glm.radians(yaw)
	pitch = glm.radians(pitch)
	roll = glm.radians(roll)
	ys, yc = sin(yaw), cos(yaw)
	ps, pc = sin(pitch), cos(pitch)
	rs, rc = sin(roll), cos(roll)
	ym = mat3(
		yc, -ys, 0.0,
		ys, yc, 0.0,
		0.0, 0.0, 1.0,
	)
	pm = mat3(
		1.0, 0.0, 0.0,
		0.0, pc, -ps,
		0.0, ps, pc,
	)
	rm = mat3(
		rc, 0.0, rs,
		0.0, 1.0, 0.0,
		-rs, 0.0, rc,
	)
	return rm * pm * ym

def make_ortho_matrix():
	return mat4(
		1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	)

def make_projection_matrix(distance: float, near: float):
	return mat4(
		distance, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 1.0,
		0.0, distance, 0.0, 0.0,
		0.0, 0.0, -2.0 * near, 0.0,
	)

def make_rescale_matrix(width: float, height: float):
	return mat4(
		1.0 / width, 0.0, 0.0, 0.0,
		0.0, 1.0 / height, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 1.0,
	)

def read_file(name) -> bytes:
	with open(os.path.join(app_root, name), 'rb') as f:
		return f.read()

def compile_shader(mode, code):
	shader = glCreateShader(mode)
	glShaderSource(shader, code)
	glCompileShader(shader)
	if not glGetShaderiv(shader, GL_COMPILE_STATUS):
		raise Error('Shader compilation error: %s\n' % glGetShaderInfoLog(shader))
	return shader

def link_program(*shaders):
	program = glCreateProgram()
	for shader in shaders:
		glAttachShader(program, shader)
	glLinkProgram(program)
	for shader in shaders:
		glDeleteShader(shader)
	if not glGetProgramiv(program, GL_LINK_STATUS):
		raise Error('Shader compilation error: %s\n' % glGetProgramInfoLog(program))
	return program

def load_mesh(name, parts = 2):
	mesh = []
	for part in range(parts):
		index_data = read_file(f'{name}.{part}.i')
		vertex_data = read_file(f'{name}.{part}.v')
		icount = len(index_data) // 2
		vcount = len(vertex_data) // 32
		ibuf, vbuf = glGenBuffers(2)
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ibuf)
		glBufferStorage(GL_SHADER_STORAGE_BUFFER, len(index_data), index_data, 0)
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbuf)
		glBufferStorage(GL_SHADER_STORAGE_BUFFER, len(vertex_data), vertex_data, 0)
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
		mesh.append((vbuf, ibuf, icount))
	return mesh

meshes = {}

def init():
	global program_mesh, program_bill, thing
	program_mesh = link_program(
		compile_shader(GL_VERTEX_SHADER, read_file("mesh.v.glsl")),
		compile_shader(GL_FRAGMENT_SHADER, read_file("mesh.f.glsl")),
	)
	program_bill = link_program(
		compile_shader(GL_VERTEX_SHADER, read_file("bill.v.glsl")),
		compile_shader(GL_GEOMETRY_SHADER, read_file("bill.g.glsl")),
		compile_shader(GL_FRAGMENT_SHADER, read_file("bill.f.glsl")),
	)
	glEnable(GL_BLEND)
	glEnable(GL_FRAMEBUFFER_SRGB)
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_LINE_SMOOTH)
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
	thing = make_thing()
	for kind in 'maple', 'pine', 'willow':
		meshes[kind] = load_mesh(kind, 2)
	prerender_bills()
	prepare_instances()
	glClearColor(0.0, 0.0, 0.2, 1.0)
	glEnable(GL_MULTISAMPLE)

if TRULY_ISOMETRIC:
	rotation_ypr = vec3(135.0, -35.26, 0.0)
	position = vec3(1.0, 1.0, 1.0)
else:
	rotation_ypr = vec3(135.0, -30.0, 0.0)
	position = vec3(1.0, 1.0, 0.816) # sqrt(2)⋅tan(30°) = sqrt(2/3)

def update(dt: float):
	global position, rotation_ypr
	vel = 3.0
	rvel = 90.0
	if FLY_CONTROLS:
		if glfw.get_key(window, glfw.KEY_LEFT): position -= dt * vel * move_matrix[0]
		if glfw.get_key(window, glfw.KEY_RIGHT): position += dt * vel * move_matrix[0]
		if glfw.get_key(window, glfw.KEY_UP): position += dt * vel * move_matrix[1]
		if glfw.get_key(window, glfw.KEY_DOWN): position -= dt * vel * move_matrix[1]
		if glfw.get_key(window, glfw.KEY_A): rotation_ypr.x += dt * rvel
		if glfw.get_key(window, glfw.KEY_D): rotation_ypr.x -= dt * rvel
		if glfw.get_key(window, glfw.KEY_W): rotation_ypr.y += dt * rvel
		if glfw.get_key(window, glfw.KEY_S): rotation_ypr.y -= dt * rvel
		if glfw.get_key(window, glfw.KEY_E): rotation_ypr.z += dt * rvel
		if glfw.get_key(window, glfw.KEY_Q): rotation_ypr.z -= dt * rvel
	else:
		if glfw.get_key(window, glfw.KEY_D): position += dt * vel * move_matrix[0]
		if glfw.get_key(window, glfw.KEY_A): position -= dt * vel * move_matrix[0]
		if glfw.get_key(window, glfw.KEY_W): position += dt * vel * move_matrix[1]
		if glfw.get_key(window, glfw.KEY_S): position -= dt * vel * move_matrix[1]
		if glfw.get_key(window, glfw.KEY_R): position += dt * vel * move_matrix[2]
		if glfw.get_key(window, glfw.KEY_F): position -= dt * vel * move_matrix[2]
		if glfw.get_key(window, glfw.KEY_SPACE): position += dt * vel * move_matrix[2]
		if glfw.get_key(window, glfw.KEY_LEFT_SHIFT): position -= dt * vel * move_matrix[2]
		if glfw.get_key(window, glfw.KEY_LEFT): rotation_ypr.x += dt * rvel
		if glfw.get_key(window, glfw.KEY_RIGHT): rotation_ypr.x -= dt * rvel
		if glfw.get_key(window, glfw.KEY_UP): rotation_ypr.y += dt * rvel
		if glfw.get_key(window, glfw.KEY_DOWN): rotation_ypr.y -= dt * rvel
	rotation_ypr.y = glm.clamp(rotation_ypr.y, -80.0, 80.0)
	rotation_ypr.z = glm.clamp(rotation_ypr.z, -60.0, 60.0)

def update0():
	global camera_matrix, move_matrix
	camera_matrix = glm.translate(mat4(make_camera_orientation_matrix(rotation_ypr.x, rotation_ypr.y, rotation_ypr.z)), -position)
	move_matrix = glm.transpose(make_camera_orientation_matrix(rotation_ypr.x, 0.0, 0.0))

def render_tripod():
	glLineWidth(1.5)
	glBegin(GL_LINES)
	glColor3f(1.0, 0.0, 0.0)
	glVertex3f(0.0, 0.0, 0.0)
	glVertex3f(1.0, 0.0, 0.0)
	glColor3f(0.0, 1.0, 0.0)
	glVertex3f(0.0, 0.0, 0.0)
	glVertex3f(0.0, 1.0, 0.0)
	glColor3f(0.0, 0.0, 1.0)
	glVertex3f(0.0, 0.0, 0.0)
	glVertex3f(0.0, 0.0, 1.0)

	glColor3f(0.0, 1.0, 1.0)
	glVertex3f(0.0, 0.0, 0.0)
	glVertex3f(-1.0, 0.0, 0.0)
	glColor3f(1.0, 0.0, 1.0)
	glVertex3f(0.0, 0.0, 0.0)
	glVertex3f(0.0, -1.0, 0.0)
	glColor3f(1.0, 1.0, 0.0)
	glVertex3f(0.0, 0.0, 0.0)
	glVertex3f(0.0, 0.0, -1.0)
	glEnd()

def make_thing(size = 1.0, branches = 128, seed = 0):
	rgen = random.Random(seed)
	da = lambda: rgen.uniform(-math.pi, math.pi)
	ddr = lambda: rgen.uniform(size / 128.0, size / 32.0)
	ddz = lambda: size / rgen.uniform(16.0, 32.0)
	pbuf = []
	cbuf = []
	ibuf = []
	for k in range(branches):
		a = da()
		dir = vec2(math.cos(a), math.sin(a))
		pos2 = vec2()
		delta = vec2(ddr(), ddz())
		blen = 0.0
		while delta.y > -delta.x:
			color = vec3(0.4 - 0.5 * blen, 0.6, 0.3 - 1.0 * blen)
			pos = vec3(pos2.x * dir, pos2.y)
			ibuf.append(len(pbuf))
			pbuf.append(pos)
			cbuf.append(color)
			pos2 += delta
			blen += glm.length(delta)
			delta.y -= 1.0 / 256.0
		ibuf.append(65535)
	ibuf.pop()
	return ((np.array(pbuf, dtype='float32'), np.array(cbuf, dtype='float32')), np.array(ibuf, dtype='uint16'))

def render_mesh(mesh):
	glLineWidth(2.5)
	glUseProgram(program_mesh)
	glUniformMatrix4fv(0, 1, GL_FALSE, projection_matrix * camera_matrix)
	#glUniformMatrix4fv(1, 1, GL_FALSE, camera_matrix)
	bufs, ibuf = mesh
	for k, buf in enumerate(bufs):
		glVertexAttribPointer(k, buf.shape[1], GL_FLOAT, GL_FALSE, 0, buf)
		glEnableVertexAttribArray(k)
	glEnable(GL_PRIMITIVE_RESTART)
	glPrimitiveRestartIndex(65535)
	glDrawElements(GL_LINE_STRIP, len(ibuf), GL_UNSIGNED_SHORT, ibuf)
	for k, buf in enumerate(bufs):
		glDisableVertexAttribArray(k)

def render_thing():
	render_mesh(thing)

def prepare_instances(gsize = 25):
	global obuf, ocount
	n = 2 * gsize + 1
	verts = np.stack([
		np.broadcast_to(np.linspace(-100.0, 100.0, n), (n, n)),
		np.broadcast_to(np.linspace(-100.0, 100.0, n), (n, n)).transpose(),
		np.zeros((n, n)),
		np.ones((n, n)),
		], axis=2).reshape((-1, 4))
	rng = rnd.default_rng()
	rng.shuffle(verts)
	ocount = len(verts)
	verts = np.array(verts, dtype='float32')
	obuf = glGenBuffers(1)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, obuf)
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, len(verts) * 16, verts, 0)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

model_matrix = make_ortho_matrix()

def render():
	time = glfw.get_time()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_MODELVIEW)
	glLoadMatrixf(camera_matrix)
	render_tripod()

	n = ocount
	m = n // 50

	glUseProgram(program_mesh)
	glUniformMatrix4fv(0, 1, GL_FALSE, projection_matrix * camera_matrix)
	glUniformMatrix4fv(1, 1, GL_FALSE, model_matrix)

	colors = [(0.5, 0.5, 0.5), (0.3, 0.5, 0.0)]

	glEnableVertexAttribArray(0)
	glEnableVertexAttribArray(2)
	glEnableVertexAttribArray(3)
	glVertexAttribDivisor(2, 1)

	for (vbuf, ibuf, count), color in zip(meshes['maple'], colors):
		glVertexAttrib3f(1, *color)

		glBindBuffer(GL_ARRAY_BUFFER, vbuf)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

		glBindBuffer(GL_ARRAY_BUFFER, obuf)
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

		glBindBuffer(GL_ARRAY_BUFFER, vbuf)
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
		glBindBuffer(GL_ARRAY_BUFFER, 0)

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf)
		glDrawElementsInstanced(GL_TRIANGLES, count, GL_UNSIGNED_SHORT, ctypes.c_void_p(0), m)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

	glVertexAttribDivisor(2, 0)
	glDisableVertexAttribArray(0)
	glDisableVertexAttribArray(2)
	glDisableVertexAttribArray(3)

	#glUseProgram(0)
	#glColor3f(0.0, 0.1, 0.0)
	#glBegin(GL_QUADS)
	#glVertex2f(-1000.0, -1000.0)
	#glVertex2f(-1000.0,  1000.0)
	#glVertex2f( 1000.0,  1000.0)
	#glVertex2f( 1000.0, -1000.0)
	#glEnd()

	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
	glUseProgram(program_bill)
	glUniformMatrix4fv(0, 1, GL_FALSE, projection_matrix)
	glUniformMatrix3fv(1, 1, GL_FALSE, mat3(camera_matrix))
	glUniform3fv(2, 1, position)
	glBindTextureUnit(0, 1)
	glVertexAttrib2f(1, 1.0, 1.0)
	glUniform1i(3, angles)
	glEnableVertexAttribArray(0)
	glBindBuffer(GL_ARRAY_BUFFER, obuf)
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0)
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glDrawArrays(GL_POINTS, m, n - m)
	glDisableVertexAttribArray(0)
	glUseProgram(0)
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

def prerender_bills():
	global projection_matrix, camera_matrix

	glClearColor(0.0, 0.0, 0.0, 0.0) #обязательно α=0

	projection_matrix = make_ortho_matrix()

	levels = 10
	size = 1 << levels
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 1)

	glBindTexture(GL_TEXTURE_2D_ARRAY, 1)
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, GL_RGBA16F, size, size, angles // 2 + 1)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY, 8.0)

	glBindTexture(GL_TEXTURE_2D_ARRAY, 2)
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, GL_DEPTH_COMPONENT16, size, size, angles // 2 + 1)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY, 8.0)

	glVertexAttrib3f(2, 0, 0, 0)
	glViewport(0, 0, size, size)
	for view in range(angles // 2 + 1):
		camera_matrix = glm.translate(glm.rotate(mat4(1.0), float((0.5 - 1.0 * view / angles) * math.pi), vec3(1.0, 0.0, 0.0)), vec3(0.0, 0.0, -0.5))
		glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 1, 0, view)
		glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 2, 0, view)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		render_thing()

	glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, [GL_DEPTH_ATTACHMENT])
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
	glGenerateTextureMipmap(1)
	glGenerateTextureMipmap(2)

def resize_window(wnd, width: int, height: int):
	global projection_matrix
	m = min(width, height)
	glViewport(0, 0, width, height)
	projection_matrix = make_rescale_matrix(width / m, height / m) * make_projection_matrix(2.0, 0.1)
	glMatrixMode(GL_PROJECTION)
	glLoadMatrixf(projection_matrix)

#void APIENTRY debug(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const *message, void const *userParam) {
	#std::printf("%.*s\n", (int)length, message)
#}

def main():
	global window
	glfw.init()
	glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
	glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
	glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
	glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
	glfw.window_hint(glfw.CONTEXT_ROBUSTNESS, glfw.LOSE_CONTEXT_ON_RESET)
	glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, True)
	glfw.window_hint(glfw.SAMPLES, 4)
	window = glfw.create_window(1024, 768, "Forest", None, None)
	glfw.make_context_current(window)
	#glDebugMessageCallback(debug, nullptr)
	init()
	#resize_window(1024, 768)
	glfw.set_window_size_callback(window, resize_window)
	t0 = glfw.get_time()
	while not glfw.window_should_close(window):
		glfw.poll_events()
		update0()
		render()
		glfw.swap_buffers(window)
		t1 = glfw.get_time()
		update(t1 - t0)
		t0 = t1
	glfw.destroy_window(window)
	glfw.terminate()

if __name__ == "__main__":
	app_root = os.path.dirname(__file__)
	main()

#import os, time
#from array import array
#import moderngl
#import moderngl_window
#from OpenGL.GL import *
#import numpy as np
#import math
#import glm
#from math import *
#from glm import *

#class TimeMeterF:
	#def __init__(self, frame_count: int = 64):
		#self.running_average = 0.0
		#self._frame_id = 0
		#self._frame_times = np.full((frame_count), 1.0)
		#x = np.linspace(-1.0, 1.0, endpoint = False, num = frame_count) + 0.5 / frame_count
		#self._weights = np.exp(-4.0 * x**2)

	#def next_frame(self, frame_time: float):
		#self._frame_times[self._frame_id] = frame_time
		#self._frame_id += 1
		#self._frame_id %= len(self._frame_times)
		#self.running_average = np.average(self._frame_times, weights=self._weights)

#class TimeMeter1:
	#def __init__(self, sensitivity = 2.0):
		#self.sensitivity = sensitivity
		#self.running_average = 0.0

	#def next_frame(self, frame_time: float):
		#c = math.exp(-self.sensitivity * frame_time)
		#self.running_average = c * self.running_average + (1 - c) * frame_time

#class TimeMeter2:
	#def __init__(self, sensitivity = 8.0):
		#self.sensitivity = sensitivity
		#self.running_average = 0.0
		#self.running_average_vel = 0.0

	#def next_frame(self, dt: float):
		#a = dt - self.running_average
		#self.running_average_vel += self.sensitivity**2 * dt * a
		#c = math.exp(-2.0 * self.sensitivity * dt)
		#self.running_average_vel = c * self.running_average_vel
		#self.running_average += dt * self.running_average_vel

#def FPSMeter(time_meter_class, *args, **kwargs):
	#class FPSMeter(time_meter_class):
		#def __init__(self, *args, **kwargs):
			#super().__init__(*args, **kwargs)
			#self.running_average = 1.0

		#@property
		#def fps(self):
			#return 1.0 / self.running_average
	#return FPSMeter(*args, **kwargs)

	#def render(self, time, frame_time):
		#if frame_time > 0:
			#self.fps_meter.next_frame(frame_time)
			#self.wnd.title = f'{self.fps_meter.fps:.1f} FPS in {self.title}'
		##self.prog['mvp'].value = (
			##1, 0, 0, 0,
			##0, 0, 1, 0,
			##0, 1, 0, 0,
			##0, -0.5, 0, 1)
		##self.tree.render(moderngl.LINES)

	#stage_map = {
		#'v': 'vertex',
		#'tc': 'tess_control',
		#'te': 'tess_evaluation',
		#'g': 'geometry',
		#'f': 'fragment',
	#}

	#def build_program(self, name, *stages):
		#args = {}
		#for st in stages:
			#stname = self.stage_map[st]
			#with open(f'{app_root}/{name}.{st}.glsl', 'r') as f:
				#args[f'{stname}_shader'] = f.read()
		#return self.ctx.program(**args)
