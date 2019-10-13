from contextlib import contextmanager
from cProfile import Profile
from pstats import Stats
import time


class ProfileToFile(Profile):
    """Profiler class with __call__()able contet manager from stackoverflow"""
    def __init__(self, *args, **kwargs):
        super(Profile, self).__init__(*args, **kwargs)
        self.disable()

    @contextmanager
    def __call__(self,info, f=None):
        self.enable()
        yield
        print(info)
        self.disable()
        if type(f)==str:
            self.dumpstats(f)

class Timer(object):
    def __init__(self, *args, **kwargs):
        super(object, self).__init__(*args, **kwargs)

    @contextmanager
    def __call__(self, info):
        t = time.time()
        yield
        dt = time.time()-t
        print("Time spent in {} Timer: "%str(info), dt)
