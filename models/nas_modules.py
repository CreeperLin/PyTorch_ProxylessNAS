# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
from models.ops import OPS, Identity, DropPath_
from profile.profiler import tprof
from torch.nn.parallel._functions import Broadcast

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
        if shared_p and not NASModule.new_shared_p:
            self.pid = NASModule._param_id
        else:
            self.pid = NASModule.add_param(params_shape)
            NASModule.new_shared_p = False
        NASModule.add_module(self, self.id, self.pid)
        print('reg NAS module: {} {}'.format(self.id, self.pid))
    
    @staticmethod
    def set_device(dev_list):
        NASModule._dev_list = dev_list if len(dev_list)>0 else [None]

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
        NASModule._params_map[pid].append(mid)
    
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
    def params():
        for p in NASModule._params:
            yield p
    
    def get_param(self):
        return NASModule._params[self.pid]
    
    def state_dict(self):
        if not self.id in NASModule._module_state_dict:
            NASModule._module_state_dict[self.id] = {}
        return NASModule._module_state_dict[self.id]
    
    def get_state(self, name):
        return self.state_dict()[name]
    
    def set_state(self, name, val):
        self.state_dict()[name] = val
    
    def del_state(self, name):
        del self.state_dict()[name]
    
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
    def build_from_genotype(gene, kwargs=None):
        for m, g in zip(NASModule._modules, gene):
            m.build_from_genotype(g, **kwargs)
    
    @staticmethod
    def to_genotype(k, ops):
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
        assert self.in_deg == 1
        if self.in_deg == 1:
            self.chn_in = chn_in[0]
        else:
            self.agg_weight = nn.Parameter(torch.zeros(self.in_deg, requires_grad=True))
            self.chn_in = chn_in[0]
        self._ops = nn.ModuleList()
        for primitive in ops:
            op = OPS[primitive](self.chn_in, stride, affine=False)
            self._ops.append(op)
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
        return sum(w * op(x) for w, op in zip(w_path_f.to(device=x.device), self._ops))


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
        self._ops = nn.ModuleList()
        self.n_samples = config.samples
        self.w_lat = config.w_latency
        if not config.augment:
            self.fixed = False
            for primitive in ops:
                op = OPS[primitive](self.chn_in, stride, affine=False)
                self._ops.append(op)
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def param_forward(self, p, requires_grad=False):
        w_path = F.softmax(p, dim=-1)
        self.set_state('w_path_f', w_path.detach())
        self.set_state('s_path_f', w_path.multinomial(self.n_samples).detach())
    
    def forward(self, x):
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            assert len(x) == 1 or x.shape[1] == self.chn_in
            if len(x) == 1:
                x = x[0]
        if self.fixed:
            return self.op(x)
        dev_id = x.device.index
        dev_id = '_' if dev_id is None else str(dev_id)
        self.set_state('x_f'+dev_id, x.detach())
        smp = self.get_state('s_path_f')
        self.swap_ops(smp, x.device)
        mid = str(self.id) + '_' + str(int(smp[0]))
        tprof.timer_start(mid)
        m_out = sum(self._ops[i](x) for i in smp)
        self.set_state('m_out'+dev_id, m_out)
        tprof.timer_stop(mid)
        tprof.add_acc_item('model', mid)
        return m_out
        # torch.cuda.empty_cache()

    def swap_ops(self, samples, device):
        for i, op in enumerate(self._ops):
            if i in samples:
                op.to(device=device)
                for p in op.parameters():
                    if not p.is_leaf: continue
                    p.requires_grad = True
            else:
                op.to(device='cpu')
                for p in op.parameters():
                    if not p.is_leaf: continue
                    p.requires_grad = False
                    p.grad = None

    def param_grad(self, loss):
        dev_id = loss.device.index
        dev_id = '_' if dev_id is None else str(dev_id)
        with torch.no_grad():
            samples = range(len(self._ops))
            a_grad = torch.zeros(self.params_shape)
            m_out = self.get_state('m_out'+dev_id)
            y_grad = (torch.autograd.grad(loss, m_out, retain_graph=True)[0]).detach()
            m_out.detach_()
            x_f = self.get_state('x_f'+dev_id)
            w_path_f = self.get_state('w_path_f')
            for j in samples:
                op = self._ops[j].to(device=x_f.device)
                op_out = op(x_f)
                op_out.detach_()
                op.to(device='cpu')
                g_grad = torch.sum(torch.mul(y_grad, op_out))
                g_grad.detach_()
                mid = str(self.id) + '_' + str(int(j))
                lat_term = 0 if self.w_lat is None else tprof.avg(mid) * self.w_lat
                for i in range(self.params_shape[0]):
                    kron = 1 if i==j else 0
                    a_grad[i] += (g_grad + lat_term) * w_path_f[j] * (kron - w_path_f[i])
            a_grad.detach_()
        return a_grad
    
    def to_genotype(self, k, ops):
        # assert ops[-1] == 'none'
        w = F.softmax(self.get_param().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, 1)
        gene = [ops[i] for i in prim_idx if ops[i]!='none']
        if gene == []: return -1, None
        return w_max, gene
    
    def build_from_genotype(self, gene, drop_path=True):
        op_name = gene[0]
        op = OPS[op_name](self.chn_in, stride=1, affine=True)
        if drop_path and not isinstance(op, Identity): # Identity does not use drop path
            op = nn.Sequential(
                op,
                DropPath_()
            )
        self.op = op.to(device='cuda')
        self.fixed = True


class NASController(nn.Module):
    def __init__(self, config, criterion, ops, device_ids=None, net_cls=None, net_kwargs={}, net=None):
        super().__init__()
        self.n_samples = config.samples
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.ops = ops
        self.net = net_cls(config, **net_kwargs) if net is None else net

    def forward(self, x):
        
        NASModule.param_forward()
        tprof.begin_acc_item('model')
        
        if len(self.device_ids) <= 1:
            return self.net(x)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)

        # replicate modules
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

        logger.info("####### ALPHA #######")
        logger.info(torch.stack(tuple(F.softmax(a.detach(), dim=-1) for a in self.alphas()), dim=0))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def dags(self):
        return self.net.dags()

    def genotype(self,k=2):
        try:
            gene_dag = self.net.to_genotype(k, ops=self.ops)
            return gt.Genotype(dag=gene_dag, ops=None)
        except:
            gene_ops = NASModule.to_genotype(k=1, ops=self.ops)
            return gt.Genotype(ops=gene_ops, dag=None)
    
    def build_from_genotype(self, gene):
        try:
            self.net.build_from_genotype(gene)
        except:
            NASModule.build_from_genotype(gene)

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