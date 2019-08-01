# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class MergerBase():
    def __init__(self):
        pass

    def chn_out(self, edges):
        pass
    
    def merge(self, states):
        pass


class ConcatMerger(MergerBase):
    def __init__(self, start=0):
        super().__init__()
        self.start = start
    
    def chn_out(self, chn_states):
        return sum(chn_states[self.start:])
    
    def merge(self, states):
        return torch.cat(states[self.start:], dim=1)


class SumMerger(MergerBase):
    def __init__(self, start=0):
        super().__init__()
        self.start = start
    
    def chn_out(self, chn_states):
        return chn_states[0]
    
    def merge(self, states):
        return sum(states[self.start:])


class LastMerger(MergerBase):
    def __init__(self):
        super().__init__()
    
    def chn_out(self, chn_states):
        return chn_states[-1]
    
    def merge(self, states):
        return states[-1]


class EnumeratorBase():
    def __init__(self):
        pass
    
    def enum(self, n_states, n_inputs):
        pass

    def len_enum(self, n_states, n_inputs):
        pass


class CombinationEnumerator():
    def __init__(self):
        super().__init__()
    
    def enum(self, n_states, n_inputs):
        return itertools.combinations(range(n_states), n_inputs)

    def len_enum(self, n_states, n_inputs):
        return len(list(itertools.combinations(range(n_states), n_inputs)))


class LastNEnumerator():
    def __init__(self):
        super().__init__()
    
    def enum(self, n_states, n_inputs):
        yield [n_states-i-1 for i in range(n_inputs)]

    def len_enum(self, n_states, n_inputs):
        return 1


class AllocatorBase():
    def __init__(self):
        pass
    
    def alloc(self, states, eidx, etot):
        pass


class SplitAllocator(AllocatorBase):
    def __init__(self):
        super().__init__()
    
    def alloc(self, states, eidx, etot):
        return [s[s.shape[0]*eidx//etot : s.shape[0]*(eidx+1)//etot] for s in states]


class ReplicateAllocator(AllocatorBase):
    def __init__(self):
        super().__init__()
    
    def alloc(self, states, eidx, etot):
        return states

