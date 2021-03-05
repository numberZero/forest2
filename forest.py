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
from glm import vec2, vec3, vec4, mat2, mat3, mat4

import fps_meter
import glm_pyogl

TRULY_ISOMETRIC = False
FLY_CONTROLS = False
h_circle_steps = 64
v_halfcircle_steps = 32
levels = 8
v_step = math.pi / v_halfcircle_steps
vsteps = v_halfcircle_steps // 2 + 1
hsteps = np.array([max(1, int(h_circle_steps * math.sin(k * v_step))) for k in range(vsteps)], dtype='uint32')
hsteps_ends = np.cumsum(hsteps, dtype='uint32')
view_count = hsteps_ends[-1]

bill_threshold = 15.0

#format, pixel_size = GL_R3_G3_B2, 1
#format, pixel_size = GL_RGB5_A1, 2
#format, pixel_size = GL_RGBA8, 4
format, pixel_size = GL_SRGB8_ALPHA8, 4
#format, pixel_size = GL_RGBA16F, 8

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

def load_mesh(name, colors):
	mesh = []
	for part, color in enumerate(colors):
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
		mesh.append((vbuf, ibuf, icount, color))
	return mesh

def make_hstb():
	global hstb
	hstb = glGenBuffers(1)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, hstb)
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, 4 * len(hsteps_ends), hsteps_ends, 0)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

class Mesh:
	pass
meshes = []

def init():
	global program_mesh, program_bill, program_split
	program_mesh = link_program(
		compile_shader(GL_VERTEX_SHADER, read_file("mesh.v.glsl")),
		compile_shader(GL_FRAGMENT_SHADER, read_file("mesh.f.glsl")),
	)
	program_bill = link_program(
		compile_shader(GL_VERTEX_SHADER, read_file("bill.v.glsl")),
		compile_shader(GL_GEOMETRY_SHADER, read_file("bill.g.glsl")),
		compile_shader(GL_FRAGMENT_SHADER, read_file("bill.f.glsl")),
	)
	program_split = link_program(
		compile_shader(GL_COMPUTE_SHADER, read_file("split.c.glsl")),
	)
	glEnable(GL_BLEND)
	glEnable(GL_FRAMEBUFFER_SRGB)
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_LINE_SMOOTH)
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
	br = BillRenderer()
	br.enter(levels)
	for prob, kind, colors in [
			(0.3, 'maple', [(0.075554, 0.035952, 0.021588), (0.042570, 0.108587, 0.004716)]),
			(0.2, 'pine', [(0.174414, 0.084845, 0.052037), (0.003064, 0.081508, 0.000000)]),
			(0.5, 'willow', [(0.055724, 0.034397, 0.025506), (0.041258, 0.184065, 0.029954)]),
			]:
		mesh = load_mesh(kind, colors)
		tree = Mesh()
		tree.name = kind
		tree.prob = prob
		tree.bufs = mesh
		tree.bill = br.render(mesh)
		meshes.append(tree)
	br.leave()
	make_hstb()
	prepare_instances([tree.prob for tree in meshes], 1.5, 100)
	prepare_split()
	glClearColor(0.0, 0.0, 0.2, 1.0)
	glEnable(GL_MULTISAMPLE)
	glDepthFunc(GL_LEQUAL)

if TRULY_ISOMETRIC:
	rotation_ypr = vec3(135.0, -35.26, 0.0)
	position = vec3(1.0, 1.0, 1.0)
else:
	rotation_ypr = vec3(135.0, -30.0, 0.0)
	position = vec3(1.0, 1.0, 0.816) # sqrt(2)⋅tan(30°) = sqrt(2/3)

fpsmeter = fps_meter.FPSMeter(fps_meter.TimeMeter2)

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

	fpsmeter.next_frame(dt)
	glfw.set_window_title(window, f'{fpsmeter.fps:.1f} FPS in Forest1')

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

colors = [(0.5, 0.5, 0.5), (0.3, 0.5, 0.0)]
model_matrix = glm.scale(make_ortho_matrix(), vec3(0.5))

def prepare_instances(probs, spacing = 2.0, gsize = 25):
	global obuf, ocount, ocounts, rad
	n = 2 * gsize + 1
	rad = spacing * gsize
	dx = spacing / 2.0
	ocount = n * n
	rng = rnd.default_rng()
	xx = np.broadcast_to(np.linspace(-rad, rad, n), (n, n))
	yy = xx.transpose()
	zz = np.zeros((n, n))
	ww = rng.choice(len(probs), p=probs, size=(n, n))
	xx = xx + rng.uniform(-dx, dx, (n, n))
	yy = yy + rng.uniform(-dx, dx, (n, n))
	ocounts = np.bincount(ww.flat)
	verts = np.array(np.stack([xx, yy, zz, ww], axis=2).reshape((-1, 4)), dtype='float32')
	obuf = glGenBuffers(1)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, obuf)
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, len(verts) * 16, verts, 0)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

def prepare_split():
	global indirects, indbuf, nbuf, fbuf
	parts = len(ocounts)
	bases = np.roll(np.cumsum(ocounts), 1)
	bases[0] = 0
	indirects = np.array(np.stack([
			# near
			np.zeros(parts, dtype='uint32'), # count
			np.zeros(parts, dtype='uint32'), # instance count
			np.zeros(parts, dtype='uint32'), # first index
			np.zeros(parts, dtype='uint32'), # base vertex
			bases, # base instance
			# far
			np.zeros(parts, dtype='uint32'), # count
			np.ones(parts, dtype='uint32'), # instance count
			bases, # first
			np.zeros(parts, dtype='uint32'), # base instance
			], axis=1).reshape((parts, 9)).flat, dtype='uint32')
	indbuf, nbuf, fbuf = glGenBuffers(3)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, nbuf)
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, ocount * 16, None, 0)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, fbuf)
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, ocount * 16, None, 0)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, indbuf)
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, len(indirects) * 4, None, GL_DYNAMIC_STORAGE_BIT)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

def render():
	time = glfw.get_time()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_MODELVIEW)
	glLoadMatrixf(camera_matrix)
	render_tripod()

	glUseProgram(program_split)
	glUniform3fv(0, 1, position)
	glUniform1f(1, bill_threshold)
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indbuf)
	glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, indirects)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, obuf)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, nbuf)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fbuf)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, indbuf)
	glDispatchCompute(ocount, 1, 1)
	glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_COMMAND_BARRIER_BIT)
	#print(np.frombuffer(glGetBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, 36 * 3), dtype='uint32').reshape((-1, 9)))

	glUseProgram(program_mesh)
	glUniformMatrix4fv(0, 1, GL_FALSE, projection_matrix * camera_matrix)
	glUniformMatrix4fv(1, 1, GL_FALSE, model_matrix)

	glEnableVertexAttribArray(0)
	glEnableVertexAttribArray(2)
	glEnableVertexAttribArray(3)
	glVertexAttribDivisor(2, 1)

	glBindBuffer(GL_ARRAY_BUFFER, nbuf)
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

	for k in range(len(ocounts)):
		for vbuf, ibuf, count, color in meshes[k].bufs:
			glVertexAttrib3f(1, *color)

			glBindBuffer(GL_ARRAY_BUFFER, vbuf)
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
			glBindBuffer(GL_ARRAY_BUFFER, 0)

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf)
			glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 36 * k, ctypes.c_uint32(count))
			glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, ctypes.c_void_p(36 * k))
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

	glVertexAttribDivisor(2, 0)
	glDisableVertexAttribArray(0)
	glDisableVertexAttribArray(2)
	glDisableVertexAttribArray(3)

	glUseProgram(0)
	glColor3f(0.0, 0.1, 0.0)
	glBegin(GL_QUADS)
	glVertex2f(-rad, -rad)
	glVertex2f(-rad,  rad)
	glVertex2f( rad,  rad)
	glVertex2f( rad, -rad)
	glEnd()

	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
	glUseProgram(program_bill)
	glUniformMatrix4fv(0, 1, GL_FALSE, projection_matrix)
	glUniformMatrix3fv(1, 1, GL_FALSE, mat3(camera_matrix))
	glUniform3fv(2, 1, position)
	glVertexAttrib2f(1, 1.0, 1.0)
	glUniform1i(3, v_halfcircle_steps)
	glEnableVertexAttribArray(0)
	glBindBuffer(GL_ARRAY_BUFFER, fbuf)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hstb)
	for k in range(len(ocounts)):
		glBindTextureUnit(0, meshes[k].bill)
		glDrawArraysIndirect(GL_POINTS, ctypes.c_void_p(36 * k + 20))
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
	glDisableVertexAttribArray(0)
	glUseProgram(0)
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)

class BillRenderer:
	def enter(self, mip_levels):
		self.levels = mip_levels
		self.size = 1 << mip_levels

		self.program = program_mesh

		self.projection_matrix = make_ortho_matrix()
		self.model_matrix = model_matrix

		self.depth = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.depth)
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT16, self.size, self.size)

		self.fb = glGenFramebuffers(1)
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.fb)
		glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.depth, 0)

	def leave(self):
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
		glDeleteFramebuffers(1, [self.fb])
		glDeleteTextures([self.depth])

	def render(self, mesh):
		layers = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D_ARRAY, layers)
		#glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, format, size, size, view_count)
		glTexStorage3D(GL_TEXTURE_2D_ARRAY, self.levels, format, self.size, self.size, view_count)
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY, 8.0)
		glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_LOD_BIAS, -0.5)

		glUseProgram(self.program)
		glUniformMatrix4fv(1, 1, GL_FALSE, self.model_matrix)
		glEnableVertexAttribArray(0)
		glEnableVertexAttribArray(3)
		glVertexAttrib3f(2, 0.0, 0.0, 0.0)
		glViewport(0, 0, self.size, self.size)
		view = 0
		for v in range(vsteps):
			va = v * v_step
			h_step_count = hsteps[v]
			h_step = 2 * math.pi / h_step_count
			for u in range(h_step_count):
				ha = u * h_step
				camera_matrix = glm.translate(
					glm.rotate(
						glm.rotate(
							mat4(1.0),
							#float(0.0),
							0.5 * math.pi - va,
							vec3(1.0, 0.0, 0.0)),
						ha,
						vec3(0.0, 0.0, 1.0)),
					vec3(0.0, 0.0, -0.5))
				glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, layers, 0, view)
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
				glUniformMatrix4fv(0, 1, GL_FALSE, self.projection_matrix * camera_matrix)

				for vbuf, ibuf, count, color in mesh:
					glVertexAttrib3f(1, *color)

					glBindBuffer(GL_ARRAY_BUFFER, vbuf)
					glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
					glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
					glBindBuffer(GL_ARRAY_BUFFER, 0)

					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf)
					glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_SHORT, ctypes.c_void_p(0))
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

				view += 1
		assert(view == view_count)
		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(3)
		glUseProgram(0)

		glGenerateTextureMipmap(layers)

		image_size = self.size**2
		total_layer_size = pixel_size * image_size / 0.75 # 0.75 for mipmaps
		total_gram = view_count * total_layer_size
		print(f'{total_gram / 1024**2} MiB GPU RAM used for the layers')

		return layers

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
	glfw.window_hint(glfw.SAMPLES, 8)
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