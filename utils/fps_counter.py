import time
from collections import deque
from typing import Deque


class FPSCounter:
    def __init__(self, smoothing: int = 30):
        self.smoothing = smoothing
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.fps_history: Deque[float] = deque(maxlen=smoothing)
        self.smoothed_fps = 0.0
        self.average_fps = 0.0

    def update(self) -> float:
        now = time.time()
        instantaneous_fps = 1.0 / (now - self.last_time + 1e-8)
        self.last_time = now
        self.frame_count += 1

        self.fps_history.append(instantaneous_fps)
        self.smoothed_fps = sum(self.fps_history) / len(self.fps_history)
        self.average_fps = self.frame_count / (now - self.start_time + 1e-8)

        return self.smoothed_fps

    def get_average(self) -> float:
        return self.average_fps

    def reset(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.fps_history.clear()
        self.smoothed_fps = 0.0
        self.average_fps = 0.0
