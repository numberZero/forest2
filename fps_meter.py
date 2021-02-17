import math
import numpy as np

class TimeMeterF:
	def __init__(self, frame_count: int = 64):
		self.running_average = 0.0
		self._frame_id = 0
		self._frame_times = np.full((frame_count), 1.0)
		x = np.linspace(-1.0, 1.0, endpoint = False, num = frame_count) + 0.5 / frame_count
		self._weights = np.exp(-4.0 * x**2)

	def next_frame(self, frame_time: float):
		self._frame_times[self._frame_id] = frame_time
		self._frame_id += 1
		self._frame_id %= len(self._frame_times)
		self.running_average = np.average(self._frame_times, weights=self._weights)

class TimeMeter1:
	def __init__(self, sensitivity = 2.0):
		self.sensitivity = sensitivity
		self.running_average = 0.0

	def next_frame(self, frame_time: float):
		c = math.exp(-self.sensitivity * frame_time)
		self.running_average = c * self.running_average + (1 - c) * frame_time

class TimeMeter2:
	def __init__(self, sensitivity = 8.0):
		self.sensitivity = sensitivity
		self.running_average = 0.0
		self.running_average_vel = 0.0

	def next_frame(self, dt: float):
		a = dt - self.running_average
		self.running_average_vel += self.sensitivity**2 * dt * a
		c = math.exp(-2.0 * self.sensitivity * dt)
		self.running_average_vel = c * self.running_average_vel
		self.running_average += dt * self.running_average_vel

def FPSMeter(time_meter_class, *args, **kwargs):
	class FPSMeter(time_meter_class):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.running_average = 1.0

		@property
		def fps(self):
			return 1.0 / self.running_average
	return FPSMeter(*args, **kwargs)

