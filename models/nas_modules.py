# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
from models.ops import OPS, Identity, DropPath_
from profile.profiler import tprof
from torch.nn.parallel._functions import Broadcast
from utils import param_size, param_count

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, l)
    # l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]
    return l_copies


class NASModule(nn.Module):
    _modules = []
    _params = []
    _module_id = -1
    _module_state_dict = {}
    _param_id = -1
    _params_map = {}
    new_shared_p = False

    def __init__(self, config, params_shape, shared_p=True):
        super().__init__()
        self.id = self.get_new_id()
        if config.augment or shared_p and not NASModule.new_shared_p:
            self.pid = NASModule._param_id
        else:
            self.pid = NASModule.add_param(params_shape)
            NASModule.new_shared_p = False
        NASModule.add_module(self, self.id, self.pid)
        # print('reg NAS module: {} {}'.format(self.id, self.pid))
    
    @staticmethod
    def nasmod_state_dict():
        return {
            # '_modules': NASModule._modules,
            '_params': NASModule._params,
            # '_module_id': NASModule._module_id,
            # '_module_state_dict': NASModule._module_state_dict,
            # '_param_id': NASModule._param_id,
            # '_params_map': NASModule._params_map
        }
    
    @staticmethod
    def nasmod_load_state_dict(sd):
        assert len(sd['_params']) == NASModule._param_id + 1
        for p, sp in zip(NASModule._params, sd['_params']):
            p.data.copy_(sp)

    @staticmethod
    def set_device(dev_list):
        dev_list = dev_list if len(dev_list)>0 else [None]
        NASModule._dev_list = [NASModule.get_dev_id(d) for d in dev_list]
    
    @staticmethod
    def get_device():
        return NASModule._dev_list
    
    @staticmethod
    def get_dev_id(index):
        return '_' if index is None else str(index)

    @staticmethod
    def get_new_id():
        NASModule._module_id += 1
        return NASModule._module_id

    @staticmethod
    def add_shared_param():
        NASModule.new_shared_p = True
    
    @staticmethod
    def add_param(params_shape):
        NASModule._param_id += 1
        param = nn.Parameter(1e-3*torch.randn(params_shape))
        NASModule._params.append(param)
        NASModule._params_map[NASModule._param_id] = []
        return NASModule._param_id

    @staticmethod
    def add_module(module, mid, pid):
        NASModule._modules.append(module)
        if pid >= 0: NASModule._params_map[pid].append(mid)
    
    @staticmethod
    def param_forward(params=None):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            for mid in pmap[pid]:
                mmap[mid].param_forward(NASModule._params[pid])
    
    @staticmethod
    def forward(x):
        pass
    
    @staticmethod
    def modules():
        for m in NASModule._modules:
            yield m
    
    @staticmethod
    def param_modules():
        mmap = NASModule._modules
        pmap = NASModule._params_map
        plist = NASModule._params
        for pid in pmap:
            mlist = pmap[pid]
            for mid in mlist:
                yield (plist[pid], mmap[pid])
    
    @staticmethod
    def params_grad(loss):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = mmap[mlist[0]].param_grad(loss)
            for i in range(1,len(mlist)):
                p_grad += mmap[mlist[i]].param_grad(loss)
            yield p_grad
    
    @staticmethod
    def param_backward(loss):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = mmap[mlist[0]].param_grad(loss)
            for i in range(1,len(mlist)):
                p_grad += mmap[mlist[i]].param_grad(loss)
            NASModule._params[pid].grad = p_grad

    @staticmethod
    def param_backward_from_grad(m_grad, dev_id):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = 0
            for i in range(0,len(mlist)):
                p_grad = mmap[mlist[i]].param_grad_dev(m_grad[mlist[i]], dev_id) + p_grad
            if NASModule._params[pid].grad is None:
                NASModule._params[pid].grad = p_grad
            else:
                NASModule._params[pid].grad += p_grad
    
    @staticmethod
    def params():
        for p in NASModule._params:
            yield p
    
    @staticmethod
    def module_apply(func, **kwargs):
        return [func(m, **kwargs) for m in NASModule._modules]
    
    @staticmethod
    def module_call(func, **kwargs):
        return [getattr(m, func)(**kwargs) for m in NASModule._modules]
    
    @staticmethod
    def param_module_call(func, **kwargs):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            for mid in pmap[pid]:
                getattr(mmap[mid], func)(NASModule._params[pid], **kwargs)

    def get_param(self):
        return NASModule._params[self.pid]
    
    def nas_state_dict(self):
        if not self.id in NASModule._module_state_dict:
            NASModule._module_state_dict[self.id] = {}
        return NASModule._module_state_dict[self.id]
    
    def get_state(self, name):
        sd = self.nas_state_dict()
        if not name in sd: return
        return sd[name]
    
    def set_state(self, name, val):
        self.nas_state_dict()[name] = val
    
    def del_state(self, name):
        del self.nas_state_dict()[name]
    
    @staticmethod
    def build_from_genotype_all(gene, kwargs={}):
        if gene.ops is None: return
        assert len(NASModule._modules) == len(gene.ops)
        for m, g in zip(NASModule._modules, gene.ops):
            m.build_from_genotype(g, **kwargs)
    
    @staticmethod
    def to_genotype_all(k, ops):
        gene = []
        for m in NASModule._modules:
            w, g_module = m.to_genotype(k, ops)
            gene.append(g_module)
        return gene


class DARTSMixedOp(NASModule):
    """ Mixed operation as in DARTS """
    def __init__(self, config, chn_in, stride, ops, shared_a, gene=None):
        params_shape = (len(ops), )
        super().__init__(config, params_shape, shared_a)
        self.in_deg = len(chn_in)
        self.stride = stride
        assert self.in_deg == 1
        if self.in_deg == 1:
            self.chn_in = chn_in[0]
        else:
            self.agg_weight = nn.Parameter(torch.zeros(self.in_deg, requires_grad=True))
            self.chn_in = chn_in[0]
        if not config.augment:
            self._ops = nn.ModuleList()
            for primitive in ops:
                op = OPS[primitive](self.chn_in, stride, affine=config.affine)
                self._ops.append(op)
            self.fixed = False
        else:
            self.fixed = True
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def param_forward(self, p):
        w_path = F.softmax(p, dim=-1)
        self.set_state('w_path_f', w_path)
    
    def forward(self, x):
        w_path_f = self.get_state('w_path_f')
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            x = x[0]
        if self.fixed:
            return self.op(x)
        return sum(w * op(x) for w, op in zip(w_path_f.to(device=x.device), self._ops))
    
    def to_genotype(self, k, ops):
        # assert ops[-1] == 'none'
        if self.pid == -1: return -1, [None]
        w = F.softmax(self.get_param().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w[:-1], 1)
        gene = [gt.abbr[ops[i]] for i in prim_idx]
        if gene == []: return -1, [None]
        return w_max, gene
    
    def build_from_genotype(self, gene, drop_path=True):
        op_name = gt.deabbr[gene[0]]
        op = OPS[op_name](self.chn_in, stride=self.stride, affine=True)
        if drop_path and not isinstance(op, Identity): # Identity does not use drop path
            op = nn.Sequential(
                op,
                DropPath_()
            )
        self.op = op
        self.fixed = True
        print("DARTSMixedOp: chn_in:{} stride:{} op:{} #p:{:.6f}".format(self.chn_in, self.stride, op_name, param_count(self)))


class BinGateMixedOp(NASModule):
    """ Mixed operation controlled by binary gate """
    def __init__(self, config, chn_in, stride, ops, shared_a):
        params_shape = (len(ops), )
        super().__init__(config, params_shape, shared_a)
        self.in_deg = len(chn_in)
        assert self.in_deg == 1
        if self.in_deg == 1:
            self.chn_in = chn_in[0]
        else:
            self.agg_weight = nn.Parameter(torch.zeros(self.in_deg, requires_grad=True))
            self.chn_in = chn_in[0]
        self.n_samples = config.samples
        self.w_lat = config.w_latency
        self.stride = stride
        if not config.augment:
            self._ops = nn.ModuleList()
            self.fixed = False
            for primitive in ops:
                op = OPS[primitive](self.chn_in, stride, affine=config.affine)
                self._ops.append(op)
            self.reset_ops()
            # print("BinGateMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))
        else:
            self.fixed = True
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def param_forward(self, p, requires_grad=False):
        s_op = self.get_state('s_op')
        w_path = F.softmax(p.index_select(-1, s_op), dim=-1)
        self.set_state('w_path_f', w_path.detach())
        self.set_state('s_path_f', s_op.index_select(-1, w_path.multinomial(self.n_samples).detach()))
    
    def sample_ops(self, p, n_samples=0):
        s_op = F.softmax(p, dim=-1).multinomial(n_samples).detach()
        self.set_state('s_op', s_op)
    
    def reset_ops(self):
        self.set_state('s_op', torch.arange(len(self._ops), dtype=torch.long))
    
    def forward(self, x):
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            assert len(x) == 1 or x.shape[1] == self.chn_in, 'invalid x shape: {} {}'.format(self.chn_in, x.shape)
            if len(x) == 1:
                x = x[0]
        if self.fixed:
            return self.op(x)
        dev_id = NASModule.get_dev_id(x.device.index)
        self.set_state('x_f'+dev_id, x.detach())
        smp = self.get_state('s_path_f')
        self.swap_ops(smp, x.device)
        mid = str(self.id) + '_' + str(int(smp[0]))
        tprof.timer_start(mid)
        m_out = sum(self._ops[i](x) for i in smp)
        tprof.timer_stop(mid)
        self.set_state('m_out'+dev_id, m_out)
        tprof.add_acc_item('model_'+dev_id, mid)
        return m_out
        # torch.cuda.empty_cache()

    def swap_ops(self, samples, device):
        for i, op in enumerate(self._ops):
            if i in samples:
                # op.to(device=device)
                for p in op.parameters():
                    if not p.is_leaf: continue
                    p.requires_grad = True
            else:
                # op.to(device='cpu')
                for p in op.parameters():
                    if not p.is_leaf: continue
                    p.requires_grad = False
                    p.grad = None

    def param_grad(self, m_grad):
        a_grad = 0
        for dev in NASModule.get_device():
            tid = str(self.id)
            tprof.timer_start(tid)
            a_grad = self.param_grad_dev(m_grad, dev) + a_grad
            tprof.timer_stop(tid)
            tprof.print_stat(tid)
        return a_grad
    
    def param_grad_dev(self, m_grad, dev_id):
        with torch.no_grad():
            sample_ops = self.get_state('s_op')
            a_grad = torch.zeros(self.params_shape)
            m_out = self.get_state('m_out'+dev_id)
            m_out.detach_()
            # y_grad = (torch.autograd.grad(loss, m_out, retain_graph=False,only_inputs=False)[0]).detach()
            x_f = self.get_state('x_f'+dev_id)
            w_path_f = self.get_state('w_path_f')
            s_path_f = self.get_state('s_path_f')
            for j, oj in enumerate(sample_ops):
                if oj in s_path_f:
                    op_out = m_out
                else:
                    op = self._ops[oj].to(device=x_f.device)
                    op_out = op(x_f)
                    op_out.detach_()
                    # op.to(device='cpu')
                g_grad = torch.sum(torch.mul(m_grad, op_out))
                g_grad.detach_()
                mid = str(self.id) + '_' + str(int(oj))
                lat_term = 0 if self.w_lat == 0 else tprof.avg(mid) * self.w_lat
                for i, oi in enumerate(sample_ops):
                    kron = 1 if i==j else 0
                    a_grad[oi] += (g_grad + lat_term) * w_path_f[j] * (kron - w_path_f[i])
            a_grad.detach_()
        return a_grad
    
    def to_genotype(self, k, ops):
        # assert ops[-1] == 'none'
        if self.pid == -1: return -1, [None]
        w = F.softmax(self.get_param().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, 1)
        gene = [gt.abbr[ops[i]] for i in prim_idx]
        if gene == []: return -1, [None]
        return w_max, gene
    
    def build_from_genotype(self, gene, drop_path=True):
        op_name = gt.deabbr[gene[0]]
        op = OPS[op_name](self.chn_in, stride=self.stride, affine=True)
        if drop_path and not isinstance(op, Identity): # Identity does not use drop path
            op = nn.Sequential(
                op,
                DropPath_()
            )
        self.op = op
        self.fixed = True
        print("BinGateMixedOp: chn_in:{} stride:{} op:{} #p:{:.6f}".format(self.chn_in, self.stride, op_name, param_count(self)))


class NASController(nn.Module):
    def __init__(self, config, net, criterion, ops, device_ids=None):
        super().__init__()
        self.criterion = criterion
        self.augment = config.augment
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.ops = ops
        self.net = net

    def forward(self, x):
        
        if not self.augment: NASModule.param_forward()
        
        for dev_id in NASModule.get_device():
            tprof.begin_acc_item('model_'+dev_id)
        
        if len(self.device_ids) <= 1:
            return self.net(x)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)

        # replicate modules
        self.net.to(device=self.device_ids[0])
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(xs),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        alphas = torch.stack(tuple(F.softmax(a.detach(), dim=-1) for a in self.alphas()), dim=0)
        logger.info("ALPHA: {}".format(alphas.shape))
        logger.info(alphas)

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def dags(self):
        if not hasattr(self.net,'dags'): return []
        return self.net.dags()

    def genotype(self):
        try:
            gene_dag = self.net.to_genotype(ops=self.ops)
            return gt.Genotype(dag=gene_dag, ops=None)
        except:
            gene_ops = NASModule.to_genotype_all(k=1, ops=self.ops)
            return gt.Genotype(ops=gene_ops, dag=None)
    
    def build_from_genotype(self, gene):
        try:
            self.net.build_from_genotype(gene)
        except Exception as e:
            print('failed building net from genotype: '+str(e))
            NASModule.build_from_genotype_all(gene)

    def weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield p

    def named_weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield n, p

    def alphas(self):
        return NASModule.params()

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def alpha_grad(self, loss):
        return NASModule.params_grad()

    def alpha_backward(self, loss):
        NASModule.param_backward(loss)
    
    def mops(self):
        return NASModule.modules()
    
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p


class GradPseudoNet(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass