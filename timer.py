import time


class Timer:
    def __init__(self, timeout):
        self.start = time.time()
        self.time_to_timeout = timeout

    def elapsed(self):
        return time.time() - self.start

    def elapsed_since_last(self):
        res = time.time() - self.checkpoint
        return res

    def remaining(self):
        return self.time_to_timeout - self.elapsed()

    def sync_timeout(self, timeout):
        self.checkpoint = time.time()
        self.time_to_timeout = timeout
