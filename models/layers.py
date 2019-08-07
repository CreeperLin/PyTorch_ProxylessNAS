# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.ops
import genotypes as gt
from utils import param_size
from profile.profiler import tprof

edge_id = 0

class ProxylessNASLossLayer(nn.Module):
    def __init__(self, w_lat):
        super(ProxylessNASLossLayer, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.w_lat = w_lat

    def forward(self, c_pred, c_true, latency):
        ce_loss = self.ce_loss(c_pred, c_true)
        lat_loss = self.w_lat * latency
        return ce_loss + lat_loss


class PreprocLayer(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class MergeFilterLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.w_merge = nn.Parameter(torch.ones(n_in, requires_grad=True))

    def forward(self, states):
        prob = F.softmax(self.w_merge, dim=-1)
        samples = prob.multinomial(self.n_out)
        return [states[i] for i in samples]


class DAGLayer(nn.Module):
    def __init__(self, config, n_nodes, chn_in, shared_a,
                    allocator, merger_state, merger_out, enumerator, preproc, aggregate,
                    edge_cls, edge_kwargs):
        super().__init__()
        global edge_id
        self.edge_id = edge_id
        edge_id+=1
        self.n_nodes = n_nodes
        self.n_input = len(chn_in)
        self.n_states = self.n_input + self.n_nodes
        self.n_input_e = len(edge_kwargs['chn_in'])
        self.shared_a = shared_a
        self.allocator = allocator(self.n_input, self.n_nodes)
        self.merger_state = merger_state()
        self.merger_out = merger_out(start=self.n_input)
        self.merge_out_range = self.merger_out.merge_range(self.n_states)
        self.enumerator = enumerator()

        chn_states = []
        if not preproc:
            self.preprocs = None
            chn_states.extend(chn_in)
        else:
            chn_cur = edge_kwargs['chn_in'][0]
            self.preprocs = nn.ModuleList()
            for i in range(self.n_input):
                self.preprocs.append(preproc(chn_in[i], chn_cur, 1, 1, 0, False))
                chn_states.append(chn_cur)

        self.dag = nn.ModuleList()
        self.edges = []
        self.num_edges = 0
        for i in range(n_nodes):
            cur_state = self.n_input+i
            self.dag.append(nn.ModuleList())
            num_edges = self.enumerator.len_enum(cur_state, self.n_input_e)
            for sidx in self.enumerator.enum(cur_state, self.n_input_e):
                e_chn_in = self.allocator.chn_in([chn_states[s] for s in sidx], sidx, cur_state)
                edge_kwargs['chn_in'] = e_chn_in
                edge_kwargs['shared_a'] = shared_a
                e = edge_cls(**edge_kwargs)
                self.dag[i].append(e)
                self.edges.append(e)
            self.num_edges += num_edges
            chn_states.append(self.merger_state.chn_out([ei.chn_out for ei in self.dag[i]]))
        
        print('DAGLayer: etype:{} chn_in:{} #n:{} #e:{}'.format(str(edge_cls), chn_in, self.n_nodes, self.num_edges))

        if aggregate is not None:
            self.merge_filter = aggregate(n_in=self.n_input+self.n_nodes,
                                        n_out=self.n_input+self.n_nodes//2)
        else:
            self.merge_filter = None

        self.chn_out = self.merger_out.chn_out(chn_states)
        print('DAGLayer: {}'.format(param_size(self)))
        

    def forward(self, x):
        if self.preprocs is None:
            states = [st for st in x]
        else:
            states = [self.preprocs[i](x[i]) for i in range(self.n_input)]

        for nidx, edges in enumerate(self.dag):
            res = []
            eidx = 0
            n_states = self.n_input+nidx
            for sidx in self.enumerator.enum(n_states, self.n_input_e):
                res.append(edges[eidx](
                        self.allocator.alloc([states[i] for i in sidx], sidx, n_states)
                    )
                )
                eidx += 1
            s_cur = self.merger_state.merge(res)
            states.append(s_cur)
        
        states_f = states if self.merge_filter is None else self.merge_filter(states)

        out = self.merger_out.merge(states_f)
        return out
    
    def apply_edge(self, func, kwargs):
        return [func(**kwargs) for e in self.edges]
    
    def to_genotype(self, k, ops):
        gene = []
        # assert ops[-1] == 'none' # assume last PRIMITIVE is 'none'
        n_states = self.n_input
        for edges in self.dag:
            eidx = 0
            topk_genes = []
            for sidx in self.enumerator.enum(n_states, self.n_input_e):
                w_edge, g_edge_child = edges[eidx].to_genotype(k, ops)
                if w_edge < 0: continue
                g_edge = (g_edge_child, sidx, n_states)
                eidx += 1
                if len(topk_genes) < k:
                    topk_genes.append((w_edge, g_edge))
                    continue
                for i in range(len(topk_genes)):
                    w, g = topk_genes[i]
                    if w_edge > w:
                        topk_genes[i] = (w_edge, g_edge)
                        break
            n_states += 1
            gene.append([g for w, g in topk_genes])
        return 0, gene
    
    def from_genotype(self, C_in, gene, edge_kwargs):
        """ generate discrete ops from gene """
        dag = nn.ModuleList()
        for edges in gene:
            row = nn.ModuleList()
            for g_child, sidx, n_states in edges:
                e_chn_in = self.allocator.chn_in(
                    [chn_states[s] for s in sidx], sidx, n_states)
                edge_kwargs['chn_in'] = e_chn_in
                edge_kwargs['gene'] = g_child
                e = edge_cls(**edge_kwargs)
                row.append(e)
            dag.append(row)
        return dag

class NASModule(nn.Module):
    _modules = []
    _params = []
    _module_id = -1
    _param_id = -1
    _params_map = {}

    def __init__(self, config, params_shape, shared_p=True):
        super().__init__()
        self.id = self.get_new_id()
        if shared_p:
            self.pid = NASModule._param_id
        else:
            self.pid = NASModule.add_param(params_shape)
        NASModule.add_module(self, self.id, self.pid)
        # print('reg NAS module: {} {}'.format(self.id, pid))

    @staticmethod
    def get_new_id():
        NASModule._module_id += 1
        return NASModule._module_id

    @staticmethod
    def add_param(params_shape):
        NASModule._param_id += 1
        param = nn.Parameter(1e-3*torch.randn(params_shape).cuda())
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
        pmap = NASModule._params_map if params is None else params
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
            p_grad = mmap[mlist[0]].alpha_grad(loss)
            for i in range(1,len(mlist)):
                p_grad += mmap[mlist[i]].params_grad(loss)
            yield p_grad
    
    @staticmethod
    def params():
        for p in NASModule._params:
            yield p
    
    def get_param(self):
        return NASModule._params[self.pid]
    
    @staticmethod
    def param_backward(loss):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = mmap[mlist[0]].alpha_grad(loss)
            for i in range(1,len(mlist)):
                p_grad += mmap[mlist[i]].params_grad(loss)
            NASModule._params[pid].grad = p_grad
    
    @staticmethod
    def from_genotype():
        pass
    
    def to_genotype(k, ops):
        pass


class DARTSMixedOp(NASModule):
    """ Mixed operation as in DARTS """
    def __init__(self, config, chn_in, stride, ops):
        params_shape = (len(ops), )
        super().__init__(config, params_shape)
        self.in_deg = len(chn_in)
        if self.in_deg == 1:
            self.chn_in = chn_in[0]
        else:
            self.agg_weight = nn.Parameter(torch.zeros(self.in_deg, requires_grad=True))
            self.chn_in = chn_in[0]
        self._ops = nn.ModuleList()
        for primitive in ops:
            op = models.ops.OPS[primitive](self.chn_in, stride, affine=False)
            self._ops.append(op)
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def forward(self, x, w_path, s_path=None):
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            x = x[0]
        return sum(w * op(x) for w, op in zip(w_path, self._ops))


class BinGateMixedOp(NASModule):
    """ Mixed operation controlled by binary gate """
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
        self.n_samples = config.samples
        self.s_path_f = None
        self.frozen = False
        self.w_lat = config.w_latency
        if gene is None:
            self.fixed = False
            for primitive in ops:
                op = models.ops.OPS[primitive](self.chn_in, stride, affine=False)
                self._ops.append(op)
        else:
            self.from_genotype(gene, self.chn_in, ops)
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def param_forward(self, p):
        w_path = F.softmax(p, dim=-1)
        self.w_path_f = w_path.detach()
        self.s_path_f = w_path.multinomial(self.n_samples).detach()
    
    def forward(self, x):
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            assert len(x) == 1
            x = x[0]
        if self.fixed:
            return self.op(x)
        self.x_f = x.detach()
        smp = self.s_path_f
        self.swap_ops(smp)
        mid = str(self.id) + '_' + str(int(smp[0]))
        tprof.timer_start(mid)
        self.mout = sum(self._ops[i](x) for i in smp)
        tprof.timer_stop(mid)
        tprof.add_acc_item('model', mid)
        return self.mout
        # torch.cuda.empty_cache()

    def swap_ops(self, samples):
        for i, op in enumerate(self._ops):
            if i in samples:
                op.to(device='cuda')
                for p in op.parameters():
                    if not p.is_leaf: continue
                    p.requires_grad = True
            else:
                op.to(device='cpu')
                for p in op.parameters():
                    if not p.is_leaf: continue
                    p.requires_grad = False
                    p.grad = None

    def alpha_grad(self, loss):
        with torch.no_grad():
            samples = self.s_path_f if self.frozen else range(len(self._ops))
            a_grad = torch.zeros(self.params_shape).cuda()
            y_grad = (torch.autograd.grad(loss, self.mout, retain_graph=True)[0]).detach()
            self.mout.detach_()
            for j in samples:
                op = self._ops[j].to(device='cuda')
                op_out = op(self.x_f)
                op_out.detach_()
                op.to(device='cpu')
                g_grad = torch.sum(torch.mul(y_grad, op_out))
                g_grad.detach_()
                mid = str(self.id) + '_' + str(int(j))
                lat_term = tprof.avg(mid) * self.w_lat
                for i in range(self.params_shape[0]):
                    kron = 1 if i==j else 0
                    a_grad[i] += (g_grad + lat_term) * self.w_path_f[j] * (kron - self.w_path_f[i])
            a_grad.detach_()
        return a_grad
    
    def to_genotype(self, k, ops):
        # assert ops[-1] == 'none'
        w = F.softmax(self.get_param().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, 1)
        # gene = [ops[i] for i in prim_idx]
        gene = [ops[i] for i in prim_idx if ops[i]!='none']
        if gene == []: return -1, None
        return w_max, gene
    
    def from_genotype(self, gene, C_in, ops):
        op_name = gene[0]
        op = ops.OPS[op_name](C_in, stride=1, affine=True)
        if not isinstance(op, ops.Identity): # Identity does not use drop path
            op = nn.Sequential(
                op,
                ops.DropPath_()
            )
        self.op = op.to(device='cuda')
        self.fixed = True
