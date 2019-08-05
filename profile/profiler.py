# -*- coding: utf-8 -*-
import os
import time
from functools import wraps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeProfiler(object):
    def __init__(self):
        self.table = {}
        self.acc_table = {}
        self.offset = 0
        self.timer_start('ofs')
        self.timer_stop('ofs')
        self.offset = self.table['ofs'][0]
        # print('tprof: offset {}'.format(self.offset))
    
    def timer_start(self, id):
        t0 = time.clock()
        if not id in self.table:
            self.table[id] = np.array([-t0])
        else:
            arr = self.table[id]
            self.table[id] = np.append(arr, -t0)
    
    def timer_stop(self, id):
        t1 = time.clock()
        self.table[id][-1] += t1 - self.offset

    def print_stat(self, id):
        if not id in self.table: return
        arr = self.table[id]
        avg = np.mean(arr)
        tmin = np.min(arr)
        tmax = np.max(arr)
        std = np.std(arr)
        print('Time %s: %s / %s / %s / %s / %s / %s'
            % (id.center(10,' '), len(arr), format(arr[-1], '0.4f'), format(avg, '0.4f'),
            format(tmin, '0.4f'), format(tmax, '0.4f'), format(std, '0.4f')))
    
    def stat_all(self):
        for i in self.table:
            self.print_stat(i)
    
    def begin_acc_item(self, cid):
        if not cid in self.acc_table:
            self.acc_table[cid] = np.array([0.])
        else:
            arr = self.acc_table[cid]
            self.acc_table[cid] = np.append(arr, [0.])

    def add_acc_item(self, cid, id):
        arr = self.acc_table[cid]
        item = self.table[id]
        arr[-1] += item[-1]

    def clear_acc_item(self, cid):
        arr = self.acc_table[cid]
        arr.clear()

    def stat_acc(self, cid):
        arr = self.acc_table[cid]
        tsum = np.sum(arr)
        avg = np.mean(arr)
        tmin = np.min(arr)
        tmax = np.max(arr)
        std = np.std(arr)
        print('Acc Time %s : %s / %s / %s / %s / %s / %s'
            % (cid.center(10,' '), len(arr), format(arr[-1], '0.4f'), format(avg, '0.4f'),
             format(tmax, '0.4f'), format(tmin, '0.4f'), format(std, '0.4f')))

    def avg(self, id):
        return 0 if not id in self.table else np.mean(self.table[id])

tprof = TimeProfiler()

if __name__ == "__main__":
    for i in range(1):
        tprof.begin_acc_item('t')
        tprof.timer_start('test')
        time.sleep(np.random.random())
        tprof.timer_stop('test')
        tprof.add_acc_item('t','test')
        tprof.print_stat('test')
        tprof.timer_start('test')
        time.sleep(np.random.random())
        tprof.timer_stop('test')
        tprof.add_acc_item('t','test')
        tprof.print_stat('test')
        tprof.stat_acc('t')