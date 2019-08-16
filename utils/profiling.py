# -*- coding: utf-8 -*-

import sys
import time
import torch
from functools import wraps

t0 = 0
def report_time(msg=''):
    global t0
    t1 = time.clock()
    fr = sys._getframe(1)
    print ("CPU Time: {} {} @ {} : {:.3f} dt: {:.3f} sec".format(
        msg.center(20,' '), fr.f_code.co_name, fr.f_lineno, t1, t1 - t0))
    t0 = t1

m0 = 0
def report_mem(msg=''):
    global m0
    m1 = torch.cuda.memory_allocated() / 1024. / 1024.
    fr = sys._getframe(1)
    print ("GPU Mem: {} {} @ {} : {:.3f} dt: {:.3f} MB".format(
        msg.center(20,' '), fr.f_code.co_name, fr.f_lineno, m1, m1 - m0))
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
        print ("GPU Mem: %s {:.3f} / {:.3f} / {:3.f} / {:.3f} MB".format(
            fname.center(20,' '), m1, m2, fp, mtable[fname]))
        return result
    return gpu_mem_profiler


def profile_time(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.clock()
        result = function(*args, **kwargs)
        t1 = time.clock()
        lat = t1 - t0
        print ("CPU Time: {}: {:.3f} / {:.3f} / {:.3f} sec".format(
            function.__name__.center(20,' '), t0, t1, lat))
        return result
    return function_timer
