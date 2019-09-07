# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
from utils import param_size, param_count
from profile.profiler import tprof
from models.nas_modules import NASModule

edge_id = 0

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
    def __init__(self, config, n_nodes, chn_in, stride, shared_a,
                    allocator, merger_state, merger_out, enumerator, preproc, aggregate,
                    edge_cls, edge_kwargs):
        super().__init__()
        global edge_id
        self.edge_id = edge_id
        edge_id+=1
        self.n_nodes = n_nodes
        self.chn_in = chn_in
        self.n_input = len(chn_in)
        self.n_states = self.n_input + self.n_nodes
        self.n_input_e = len(edge_kwargs['chn_in'])
        self.shared_a = shared_a
        if shared_a:
            NASModule.add_shared_param()
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

        if not config.augment:
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
                    edge_kwargs['stride'] = stride if all(s < self.n_input for s in sidx) else 1
                    edge_kwargs['shared_a'] = shared_a
                    e = edge_cls(**edge_kwargs)
                    self.dag[i].append(e)
                    self.edges.append(e)
                self.num_edges += num_edges
                chn_states.append(self.merger_state.chn_out([ei.chn_out for ei in self.dag[i]]))
                self.chn_out = self.merger_out.chn_out(chn_states)
            print('DAGLayer: etype:{} chn_in:{} #n:{} #e:{}'.format(str(edge_cls), self.chn_in, self.n_nodes, self.num_edges))
            print('DAGLayer param count: {:.6f}'.format(param_count(self)))
        else:
            self.chn_states = chn_states
            self.edge_cls = edge_cls
            self.edge_kwargs = edge_kwargs

        if aggregate is not None:
            self.merge_filter = aggregate(n_in=self.n_input+self.n_nodes,
                                        n_out=self.n_input+self.n_nodes//2)
        else:
            self.merge_filter = None
        self.chn_out = self.merger_out.chn_out(chn_states)

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
                e_in = self.allocator.alloc([states[i] for i in sidx], sidx, n_states)
                res.append(edges[eidx](e_in))
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
    
    def build_from_genotype(self, gene):
        """ generate discrete ops from gene """
        self.dag = nn.ModuleList()
        chn_states = self.chn_states
        edge_cls = self.edge_cls
        edge_kwargs = self.edge_kwargs
        num_edges = 0
        for edges in gene:
            row = nn.ModuleList()
            for g_child, sidx, n_states in edges:
                e_chn_in = self.allocator.chn_in(
                    [chn_states[s] for s in sidx], sidx, n_states)
                edge_kwargs['chn_in'] = e_chn_in
                e = edge_cls(**edge_kwargs)
                e.build_from_genotype(g_child)
                row.append(e)
                num_edges += 1
            self.dag.append(row)
            chn_states.append(self.merger_state.chn_out([ei.chn_out for ei in row]))
        self.num_edges = num_edges
        self.chn_states = chn_states
        self.chn_out = self.merger_out.chn_out(chn_states)
        print('DAGLayer: etype:{} chn_in:{} #n:{} #e:{}'.format(str(edge_cls), self.chn_in, self.n_nodes, self.num_edges))
        print('DAGLayer param count: {:.6f}'.format(param_count(self)))


class TreeLayer(nn.Module):
    def __init__(self, config, n_nodes, chn_in, stride, shared_a,
                    allocator, merger_out, preproc, aggregate,
                    child_cls, child_kwargs, edge_cls, edge_kwargs,
                    children=None, edges=None):
        super().__init__()
        self.edges = nn.ModuleList()
        self.subnets = nn.ModuleList()
        chn_in = (chn_in, ) if isinstance(chn_in, int) else chn_in
        self.n_input = len(chn_in)
        self.n_nodes = n_nodes
        self.n_states = self.n_input + self.n_nodes
        self.allocator = allocator(self.n_input, self.n_nodes)
        self.merger_out = merger_out(start=self.n_input)
        self.merge_out_range = self.merger_out.merge_range(self.n_states)
        if shared_a:
            NASModule.add_shared_param()

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
        
        sidx = range(self.n_input)
        for i in range(self.n_nodes):
            e_chn_in = self.allocator.chn_in([chn_states[s] for s in sidx], sidx, i)
            if not edges is None:
                self.edges.append(edges[i])
                c_chn_in = edges[i].chn_out
            elif not edge_cls is None:
                edge_kwargs['chn_in'] = e_chn_in
                edge_kwargs['stride'] = stride
                if 'shared_a' in edge_kwargs: edge_kwargs['shared_a'] = shared_a
                e = edge_cls(**edge_kwargs)
                self.edges.append(e)
                c_chn_in = e.chn_out
            else:
                self.edges.append(None)
                c_chn_in = e_chn_in
            if not children is None:
                self.subnets.append(children[i])
            elif not child_cls is None:
                child_kwargs['chn_in'] = c_chn_in
                self.subnets.append(child_cls(**child_kwargs))
            else:
                self.subnets.append(None)
        
        if aggregate is not None:
            self.merge_filter = aggregate(n_in=self.n_states,
                                        n_out=self.n_states//2)
        else:
            self.merge_filter = None
        
        print('TreeLayer: etype:{} ctype:{} chn_in:{} #node:{} #p:{:.6f}'.format(str(edge_cls), str(child_cls), chn_in, self.n_nodes, param_count(self)))
    
    def forward(self, x):
        x = [x] if not isinstance(x, list) else x
        if self.preprocs is None:
            states = [st for st in x]
        else:
            states = [self.preprocs[i](x[i]) for i in range(self.n_input)]

        n_states = self.n_input
        sidx = range(self.n_input)
        for edge, child in zip(self.edges, self.subnets):
            out = self.allocator.alloc([states[i] for i in sidx], sidx, n_states)
            if not edge is None:
                out = edge(out)
            if not child is None:
                out = child([out])
            states.append(out)
        
        states_f = states if self.merge_filter is None else self.merge_filter(states)

        out = self.merger_out.merge(states_f)
        return out
    
    def build_from_genotype(self, gene):
        pass