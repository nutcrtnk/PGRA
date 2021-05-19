import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.functional import *
import math
from model.scoring import Matcher

def mean_pooling(x, mask, dim=-2):
    mask = mask.float()
    mask = mask / mask.sum(dim=-1, keepdim=True) + 1e-12
    x = (x * mask.unsqueeze(-1)).sum(dim)
    return x

def sum_pooling(x, mask, dim=-2):
    mask = mask.float()
    x = (x * mask.unsqueeze(-1)).sum(dim)
    return x

class GRUMix(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.weight_ih = nn.Linear(emb_size, emb_size*3, bias=True)
        self.weight_hh = nn.Linear(emb_size, emb_size*3, bias=False)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.weight_ih.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.weight_ih.bias)

    def forward(self, self_vector, nb_vector):
        ri, zi, hi = self.weight_ih(nb_vector).chunk(3, dim=-1)
        rh, zh, hh = self.weight_hh(self_vector).chunk(3, dim=-1)
        r = self.sigm(ri + rh)
        z = self.sigm(zi + zh)
        h = self.tanh(hi + hh * r)
        return (1-z) * self_vector + z * h


class PGRAAgg(nn.Module):
    
    def __init__(self, input_size, emb_size, n_head=1, agg_act='lrelu', use_weight=False, nb_reg_all=True, nb_reg=0., nb_reg_att=True, nb_reg_mode='l2', i_hop=0, n_hop=1, detach_att=True, gru=1, att_mode='node_rela', **kwargs):
        super().__init__()
        self.n_head = n_head
        self.head_dim = emb_size // self.n_head
        if use_weight:
            self.att_w = nn.Linear(input_size, emb_size)
        else:
            self.att_w = None
        self.att_a_self = nn.Parameter(torch.zeros([self.n_head,  self.head_dim]))
        self.att_a_nb = nn.Parameter(torch.zeros([self.n_head,  self.head_dim]))
        self.agg_act = get_activation(agg_act)
        self.is_last = n_hop == i_hop
        self.i_hop = i_hop
        if self.i_hop <= gru:
            self.mixer = GRUMix(emb_size)
        else:
            self.mixer = lambda x, y: y
        if not nb_reg_all and not self.is_last:
            nb_reg = 0
        self.nb_reg = nb_reg
        self.nb_reg_att = nb_reg_att
        self.detach_att = detach_att
        self.register_buffer('inner_loss', torch.tensor(0.))
        self.att_mode = att_mode
        if self.nb_reg > 0:
            self.nb_scoring = Matcher(mode=nb_reg_mode, use_norm=False)
        self.reset_parameters()
    
    def __repr__(self):
        return '{} (heads={}, nb_reg={}, detach_att={}, nb_reg_att={}, att_mode={})'.format(
            super().__repr__(), self.n_head, self.nb_reg, self.detach_att, self.nb_reg_att, self.att_mode)

    def reset_parameters(self):
        if self.att_w is not None:
            nn.init.xavier_uniform_(self.att_w.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_a_nb, gain=1.414)
        nn.init.xavier_uniform_(self.att_a_self, gain=1.414)
        
    def forward(self, self_vector, neighbor_vectors, target_relation, neighbor_relations, relation_similarity, mask, **kwargs):
        ret_shape = self_vector.shape
        if self.att_w is not None:
            self_vector = self.att_w(self_vector)
            neighbor_vectors = self.att_w(neighbor_vectors)
        head_shape = [*self_vector.shape[:-1], 1, self.n_head, self.head_dim]
        neighbor_vectors = neighbor_vectors.view(*neighbor_vectors.shape[:-1], self.n_head, self.head_dim)
        
        self_head = self_vector.view(*head_shape)
        nb_head = neighbor_vectors
        if self.detach_att:
            self_head = self_head.detach()
            nb_head = nb_head.detach()
            if relation_similarity is not None:
                relation_similarity = relation_similarity.detach() 
        att_feat_self = (self_head * self.att_a_self).sum(-1)
        att_feat_nb = (nb_head * self.att_a_nb).sum(-1) + att_feat_self
        if 'rela' in self.att_mode and target_relation is not None and relation_similarity is not None:
            att_rela_nb = torch.gather(relation_similarity[target_relation], dim=-1, index=neighbor_relations).unsqueeze(-1)
        else:
            att_rela_nb = 1
        if self.att_mode != 'node':
            att_feat_nb = att_feat_nb + 1
        att_nb = self.agg_act(att_feat_nb)
        if 'node' not in self.att_mode:
            att_nb = att_nb.fill_(1)
        att_nb = att_nb * att_rela_nb 
        att_nb = att_nb.masked_fill(~mask.unsqueeze(-1), -float('inf'))
        att = F.softmax(att_nb, dim=-2)
        att = att.masked_fill(~mask.unsqueeze(-1), 0)
        nb_vector = att.unsqueeze(-1) * neighbor_vectors
        nb_vector = nb_vector.sum(-3)
        x = self.mixer(self_vector, nb_vector.view(*ret_shape))
        if self.training and self.nb_reg > 0:
            if self.nb_reg_att:
                att_reg = att.detach()
            else:
                att_reg = mask.float().unsqueeze(-1)
            att_reg = att_reg / att_reg.sum() * self.nb_reg
            losses = -self.nb_scoring(x.view(*head_shape), neighbor_vectors).sum(-1) * att_reg
            self.inner_loss += losses.sum()
        return x


