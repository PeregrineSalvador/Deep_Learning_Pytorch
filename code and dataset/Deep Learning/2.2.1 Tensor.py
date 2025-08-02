import time
import numpy as np

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动定时器"""
        self.tick = time.time()
    
    def stop(self):
        self.times.append(time.time()-self.tick)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times)/len(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()