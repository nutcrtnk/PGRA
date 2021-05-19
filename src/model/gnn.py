import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.weight_init import weight_init, manual_init
import numpy as np
from model.module.transform import get_relation_transform
from model.module.functional import *
from model.module.conv import PGRAAgg


class GNN(nn.Module):

    def __init__(self, n_node, n_relation, emb_size, node_type, n_hop, n_neighbor, features=None, norm_node=0, drop_f=0.6, use_pre=True, feat_act='none', pre='none', post='none', act='none', rsim=True, norm=0, homo=False, self_loop=0, self_loop_coef=1., detach_rela=False, **kwargs):
        super().__init__()
        self.self_loop = self_loop
        self.n_node = n_node
        self.n_relation = n_relation
        if self_loop:
            n_relation += 1
            self.self_loop_coef = nn.Parameter(torch.ones(1) * self_loop_coef)
        self.node_type = nn.Parameter(node_type, requires_grad=False)
        n_types = node_type.max().item() + 1
    
        if features is not None:
            self.features = nn.Parameter(features, requires_grad=False)
            self.node_emb = None
            if use_pre:
                self.f_proj = nn.Linear(features.shape[-1], emb_size, bias=False)
                self.feat_act = get_activation(feat_act)
                feat_size = emb_size
            else:
                self.f_proj = None
                feat_size = features.shape[-1]
        else:
            self.features = None
            self.node_emb = nn.Embedding(n_node, emb_size)
            feat_size = emb_size

        self.feat_size = feat_size
        self.emb_size = emb_size
        self.n_hop = n_hop
        self.n_neighbor = n_neighbor

        self.norm_node = norm_node

        self.aggregators = nn.ModuleList([
            PGRAAgg(input_size=feat_size if i==0 else emb_size, emb_size=emb_size, n_relation=n_relation, n_types=n_types, i_hop=i+1, n_hop=n_hop, **kwargs) for i in range(n_hop)
        ])

        adj_node = np.ones((n_node, n_neighbor)) * -1
        adj_rela = np.ones((n_node, n_neighbor)) * -1

        self.pre = get_relation_transform(pre)(n_relation=n_relation, emb_size=emb_size)
        self.post = get_relation_transform(post)(n_relation=n_relation, emb_size=emb_size, act=None,)
        self.register_neighbors(adj_node, adj_rela)
        self.drop_f = nn.Dropout(drop_f)
        self.act = get_activation(act)
        self.prune = None
        self.prune_self = None
        self.norm = norm
        self.rsim = rsim
        self.detach_rela = detach_rela

    def register_neighbors(self, adj_node, adj_rela, cuda=False):
        adj_node = torch.LongTensor(adj_node)
        adj_rela = torch.LongTensor(adj_rela)
        self.register_buffer('adj_node', adj_node)
        self.register_buffer('adj_rela', adj_rela)

    def get_prior_emb(self, index, target_rela):
        if self.features is not None:
            feat = F.embedding(index, self.features)
            if self.f_proj is not None:
                feat = self.f_proj(feat)
                feat = self.feat_act(feat)
            emb = self.drop_f(feat)
        else:
            emb = self.node_emb(index)
        if target_rela is not None:
            emb = self.pre(emb, target_rela)
        return emb

    def reset(self):
        self.apply(weight_init)
        self.apply(manual_init)

    @torch.no_grad()
    def reset_parameters(self):
        self.constraint()

    def get_neighbors(self, nodes):
        batch_size = nodes.size(0)
        entities = [nodes.unsqueeze(-1)]
        relations = []
        masks = []

        for i in range(self.n_hop):
            n_nb = self.n_neighbor if isinstance(self.n_neighbor, int) else self.n_neighbor[i]

            _nodes = entities[i]
            neighbor_entities = F.embedding(
                _nodes, self.adj_node)[:, :, :n_nb]
            neighbor_relations = F.embedding(
                _nodes, self.adj_rela)[:, :, :n_nb]
            
            mask = (neighbor_entities >= 0)
            neighbor_entities[~mask] = torch.randint_like(neighbor_entities[~mask], self.n_node)
            neighbor_relations[~mask] = torch.randint_like(neighbor_relations[~mask], self.n_relation)

            if i > 0:
                mask = mask & masks[-1].unsqueeze(-1)
            
            masks.append(mask.view(batch_size, -1))
            entities.append(neighbor_entities.view(batch_size, -1))
            relations.append(neighbor_relations.view(batch_size, -1))
        return entities, relations, masks

    def calc_rsim(self):
        rela_w = self.pre.rela_emb
        rela_w = rela_w.unsqueeze(0).repeat(rela_w.size(0), 1, 1)
        rela_wt = rela_w.transpose(0, 1)
        if self.detach_rela:
            rela_wt = rela_wt.detach()
        rsim = torch.abs(torch.cosine_similarity(rela_w, rela_wt, dim=-1))
        if self.self_loop:
            self_mult = torch.ones_like(rsim)
            self_mult[:, -1] = self.self_loop_coef
            rsim = rsim * self_mult
        return rsim

    def aggregate(self, target_rela, entity, relation, masks, **kwargs):
        batch_size = entity[0].size(0)
        entity_vectors = [self.get_prior_emb(_entities, target_rela) for _entities in entity]
        if self.rsim:
            relation_similarity = self.calc_rsim()
        else:
            relation_similarity = None

        for i in range(self.n_hop):
            entity_vectors_next_iter = []
            for hop in range(self.n_hop - i):
                n_nb = self.n_neighbor if isinstance(self.n_neighbor, int) else self.n_neighbor[hop]
                shape = [-1, n_nb, self.emb_size if i > 0 else self.feat_size]
                self_vector = entity_vectors[hop].view(-1, self.emb_size if i > 0 else self.feat_size)
                neighbor_vectors = entity_vectors[hop + 1].view(*shape)
                if self.prune is not None:
                    neighbor_vectors = self.prune(neighbor_vectors)
                if self.prune_self is not None:
                    self_vector = self.prune_self(self_vector)
                mask = masks[hop].view(*shape[:-1])
                neighbor_relations=relation[hop].view(*shape[:-1])
                self_type = self.node_type[entity[hop]].view(-1)
                neighbor_type= self.node_type[entity[hop+1]].view(*shape[:-1])
                if target_rela is not None:
                    target_relation = torch.repeat_interleave(target_rela, self_vector.size(0) // target_rela.size(0))
                else:
                    target_relation = None
                agg_kwargs = dict(self_vector=self_vector,
                                             neighbor_vectors=neighbor_vectors,
                                             target_relation= target_relation,
                                             neighbor_relations=neighbor_relations,
                                             self_type = self_type,
                                             neighbor_type = neighbor_type,
                                             relation_similarity = relation_similarity,
                                             mask=mask)
                aggregators = self.aggregators if target_rela is not None else self.global_aggregators
                vector = aggregators[i](**agg_kwargs, **kwargs)
                if i != self.n_hop - 1:
                    vector = self.act(vector)
                entity_vectors_next_iter.append(vector)

            entity_vectors = entity_vectors_next_iter
        res = entity_vectors[0].view(batch_size, self.emb_size)
        return res

    def forward(self, node, relation, **kwargs):
        entities, nb_relation, masks = self.get_neighbors(node)
        node_emb = self.aggregate(relation, entities, nb_relation, masks, **kwargs)
        if self.post is not None:
            node_emb = self.post(node_emb, relation)
        if self.norm > 0:
            node_emb = F.normalize(node_emb, dim=-1) * self.norm
        return node_emb

    def constraint(self):
        if self.norm_node and self.node_emb is not None:
            self.node_emb.weight.data = F.normalize(self.node_emb.weight.data) * self.norm_node
        for module in self.modules():
            if module != self and hasattr(module, 'constraint'):
                module.constraint()
