import time
from collections import deque

class FPSCounter():
    def __init__(self, smoothing):
        self.timestamps = deque(maxlen = smoothing)
        self.last_time = None
        
    def update(self):
        current_time = time.time()
        if self.last_time is not None:
            fps = 1.0 / (current_time - self.last_time)
            self.timestamps.append(fps)
        self.last_time = current_time
    
    def get_fps(self):
        if not self.timestamps:
            return 0.0
        return sum(self.timestamps) / len(self.timestamps)
    
    def reset(self):
        self.timestamps.clear()
        self.last_time = None
        