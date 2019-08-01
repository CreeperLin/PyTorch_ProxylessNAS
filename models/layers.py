# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.ops
import genotypes as gt
from utils import param_size

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
        self.n_input_e = len(edge_kwargs['chn_in'])
        self.shared_a = shared_a
        self.allocator = allocator
        self.merger_state = merger_state
        self.merger_out = merger_out
        self.enumerator = enumerator

        chn_states = []
        if not preproc:
            self.preprocs = None
            chn_states.extend(edge_kwargs['chn_in'])
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
            self.dag.append(nn.ModuleList())
            num = 0
            for sidx in self.enumerator.enum(self.n_input+i, self.n_input_e):
                e_chn_in = [chn_states[s] for s in sidx]
                edge_kwargs['chn_in'] = e_chn_in
                e = edge_cls(**edge_kwargs)
                self.dag[i].append(e)
                self.edges.append(e)
                num += 1
            self.num_edges += num
            chn_states.append(self.merger_state.chn_out([ei.chn_out for ei in self.dag[i]]))
        
        # print(self.dag)
        print('DAGLayer: etype:{} chn_in:{} #n:{} #e:{}'.format(str(edge_cls), chn_in, self.n_nodes, self.num_edges))

        self.alphas_shape = e.alphas_shape
        if not self.shared_a:
            self.alphas_shape = (self.num_edges, ) + self.alphas_shape
        
        if aggregate is not None:
            self.merge_filter = aggregate(n_in=self.n_input+self.n_nodes,
                                        n_out=self.n_input+self.n_nodes//2)
        else:
            self.merge_filter = None

        self.chn_out = self.merger_out.chn_out(chn_states)
        print('DAGLayer: {}'.format(param_size(self)))
        

    def forward(self, x, w_dag):
        if self.preprocs is None:
            states = [st for st in x]
        else:
            states = [self.preprocs[i](x[i]) for i in range(self.n_input)]

        widx = 0
        for edges in self.dag:
            res = []
            eidx = 0
            for sidx in self.enumerator.enum(len(states), self.n_input_e):
                w_edge = w_dag if self.shared_a else w_dag[widx+eidx]
                res.append(edges[eidx](
                        self.allocator.alloc([states[i] for i in sidx], eidx, len(edges)),
                        w_edge
                    )
                )
                eidx += 1
            widx += eidx
            s_cur = self.merger_state.merge(res)
            states.append(s_cur)
        
        states_f = states if self.merge_filter is None else self.merge_filter(states)

        out = self.merger_out.merge(states_f)
        return out
    
    def mops(self):
        for e in self.edges:
            for m in e.mops():
                yield m
        
    def alpha_grad(self, loss):
        a_grad = torch.stack(tuple(e.alpha_grad(loss) for e in self.edges), dim=0).cuda()
        if self.shared_a:
            a_grad = torch.sum(a_grad,dim=0)
        return a_grad

    def freeze(self, freeze, w_dag=None, kwargs={}):
        eidx = 0
        for edges in self.dag:
            for e in edges:
                w = None if w_dag is None else \
                    (w_dag if self.shared_a else w_dag[eidx])
                e.freeze(freeze, w, **kwargs)
                eidx += 1
    
    def apply_edge(self, func, kwargs):
        return [func(**kwargs) for e in self.edges]
    
    def to_genotype(self, alpha, k, ops):
        gene = []
        assert ops[-1] == 'none' # assume last PRIMITIVE is 'none'

        widx = 0
        n_states = self.n_input
        for n_node in range(len(self.dag)):
            eidx = 0
            for sidx in self.enumerator.enum(n_states, self.n_input_e):
                a = alpha[0] if self.shared_a else alpha[widx+eidx]
                edge_child_g = e.parse_alpha(a, k, ops)
                edge_gene = (edge_child_g, sidx, n_states)
                gene.append(edge_gene)
                eidx+=1
            widx += eidx
            n_states += 1

            
    
    
        return gene


class DARTSMixedOp(nn.Module):
    """ Mixed operation as in DARTS """
    def __init__(self, config, chn_in, stride, ops):
        super().__init__()
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
        self.alphas_shape = (len(ops), )
        self.chn_out = self.chn_in
    
    def forward(self, x, w_dag):
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            x = x[0]
        return sum(w * op(x) for w, op in zip(w_dag, self._ops))


class BinGateMixedOp(nn.Module):
    """ Mixed operation controlled by binary gate """
    def __init__(self, config, chn_in, stride, ops):
        super().__init__()
        global edge_id
        self.edge_id = edge_id
        edge_id+=1
        self.in_deg = len(chn_in)
        assert self.in_deg == 1
        if self.in_deg == 1:
            self.chn_in = chn_in[0]
        else:
            self.agg_weight = nn.Parameter(torch.zeros(self.in_deg, requires_grad=True))
            self.chn_in = chn_in[0]
        self._ops = nn.ModuleList()
        self.n_samples = config.samples
        self.samples_f = None
        self.frozen = False
        for primitive in ops:
            op = models.ops.OPS[primitive](self.chn_in, stride, affine=False)
            self._ops.append(op)
        self.alphas_shape = (len(ops), )
        self.chn_out = self.chn_in
    
    def forward(self, x, w_dag):
        if self.in_deg != 1:
            x = torch.matmul(x, torch.sigmoid(self.agg_weight))
        else:
            assert len(x) == 1
            x = x[0]
        self.w_dag_f = w_dag.detach()
        self.x_f = x.detach()
        if self.frozen:
            samples = self.samples_f 
        else:
            samples = w_dag.multinomial(self.n_samples).detach()
            self.swap_ops(samples)
        # print(gt.PRIMITIVES_DEFAULT[int(samples)])
        self.mout = sum((self._ops[i])(x) for i in samples)
        return self.mout
        # torch.cuda.empty_cache()

    def swap_ops(self, samples):
        for i, op in enumerate(self._ops):
            if i in samples:
                op.to(device='cuda')
                for p in op.parameters():
                    p.requires_grad = True
            else:
                op.to(device='cpu')
                for p in op.parameters():
                    p.requires_grad = False
                    p.grad = None

    def freeze(self, freeze, w_dag=None, n_samples=2):
        self.frozen = freeze
        if freeze:
            self.w_dag_f = w_dag.detach()
            self.samples_f = w_dag.multinomial(n_samples)
            self.swap_ops(self.samples_f)
            # print('frozen: {}'.format(self.samples_f))
    
    def alpha_grad(self, loss):
        with torch.no_grad():
            samples = self.samples_f if self.frozen else range(len(self._ops))
            a_grad = torch.zeros(self.alphas_shape).cuda()
            y_grad = (torch.autograd.grad(loss, self.mout, retain_graph=True)[0]).detach()
            self.mout.detach_()
            for j in samples:
                op = self._ops[j].to(device='cuda')
                op_out = op(self.x_f)
                op_out.detach_()
                op.to('cpu')
                g_grad = torch.sum(torch.mul(y_grad, op_out))
                g_grad.detach_()
                for i in range(self.alphas_shape[0]):
                    kron = 1 if i==j else 0
                    a_grad[i] += g_grad * self.w_dag_f[j] * (kron - self.w_dag_f[i])
            a_grad.detach_()
        return a_grad
    
    def mops(self):
        yield self

    def to_genotype(self, alpha, k, ops):
        gene = []
        assert ops[-1] == 'none' # assume last PRIMITIVE is 'none'
        # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
        # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
        for edges in alpha:
            # edges: Tensor(n_edges, n_ops)
            edge_max, primitive_indices = torch.topk(edges, 1) # ignore 'none'
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_gene = []
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = ops[prim_idx]
                node_gene.append((prim, edge_idx.item()))

            gene.append(node_gene)

        return gene