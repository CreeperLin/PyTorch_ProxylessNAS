# -*- coding: utf-8 -*-

import sys
import time
import torch
from functools import wraps

m0 = 0
def report_mem(msg=''):
    global m0
    m1 = torch.cuda.memory_allocated() / 1024. / 1024.
    fr = sys._getframe(1)
    print ("GPU Mem: %s %s @ %s : %s dt: %s MB" 
        % (msg.center(20,' '), fr.f_code.co_name, fr.f_lineno, 
            format(m1, '0.3f'), format(m1 - m0, '0.3f')))
    m0 = m1

mtable = {}
def profile_mem(function):
    @wraps(function)
    def gpu_mem_profiler(*args, **kwargs):
        m1 = torch.cuda.memory_allocated() / 1024. / 1024.
        result = function(*args, **kwargs)
        m2 = torch.cuda.memory_allocated() / 1024. / 1024.
        fp = m2 - m1
        fname = function.__name__
        if fname in mtable:
            mtable[fname] += fp
        else:
            mtable[fname] = fp
        print ("GPU Mem: %s %s / %s / %s / %s MB" %
               (fname.center(20,' '), format(m1, '0.3f'), format(m2, '0.3f'),
                format(fp, '0.3f'), format(mtable[fname], '0.3f')))
        return result
    return gpu_mem_profiler


def profile_time(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.clock()
        result = function(*args, **kwargs)
        t1 = time.clock()
        lat = t1 - t0
        print ("CPU Time: %s: %s / %s / %s sec" %
               (function.__name__.center(20,' '), format(t0, '0.3f'), format(t1, '0.3f'), format(lat, '0.3f')))
        return result
    return function_timer
