import time
import torch
class ClockContext:
    def __init__(self, default_name="default", verbose = True):
        self.t_starts = []
        self.t_ends = []
        self.t_durations = []
        self.accumulated_time = 0
        self.enter_times = 0
        self.verbose = verbose
        self.default_name=default_name
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def __enter__(self):
        self.t_starts.append(time.time())
        self.enter_times += 1
        return self
    
    def __exit__(self, *args):
        self.t_ends.append(time.time())
        time_used = self.t_ends[-1] - self.t_starts[-1]
        self.accumulated_time += time_used
        self.t_durations.append(time_used)
        torch.cuda.synchronize()
        if self.verbose:
            print(f"{self.default_name} Time used: {time_used}")

    def __str__(self):
        mean_time = self.accumulated_time / self.enter_times if self.enter_times > 0 else -1
        return f"Total Time used: {self.accumulated_time:.4f} seconds, mean time: {mean_time:.4f} seconds"

    @property
    def avg_time(self):
        return self.accumulated_time / self.enter_times if self.enter_times > 0 else -1
    
    @property
    def total_time(self):
        return self.accumulated_time