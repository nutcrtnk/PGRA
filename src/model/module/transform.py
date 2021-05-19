import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple
from torch import Tensor
from model.module.functional import *
from model.utils.weight_init import weight_init


def get_transform(name):
    if name is None or name == 'none':
        return Iden
    name = name.lower()
    if name[:5] == 'block':
        name_splited = name.split('!')
        if len(name_splited) > 1:
            conv_size = int(name_splited[1])
            return partialclass(Block, 'Block', conv_size=conv_size)
        return Block
    name_dict = {
        'iden': Iden,
        'distmult': DistMult,
        'linear': Linear,
        'transh': TransH,
        'bin': Binomial,
        'dropbin': partialclass(Binomial, 'DropBinomial', upscale=True)
    }
    return name_dict[name]


def get_relation_transform(name):
    if name is None or name == 'none':
        return Iden
    name = name.lower()
    name_dict = {
        'transh': RelaTransH,
        'distmult': RelaDistMult,
        'distmult+': partialclass(RelaDistMult, 'RelaDistMultPlus', normalize=False),
        'bin': RelaBinomial,
        'dropbin': partialclass(RelaBinomial, 'RelaDropBinomial', upscale=True)
    }
    if name in name_dict:
        return name_dict[name]
    else:
        return partialclass(RelationTransform, f'Rel_{name}', transform=name)


class Iden(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input):
        return input[0]


class RelaTransH(nn.Module):

    def __init__(self, n_relation, emb_size, **kwargs):
        super().__init__()
        self.rela_emb = nn.Parameter(torch.zeros([n_relation, emb_size]))
        self.emb_size = emb_size
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.rela_emb, 1)
        self.constraint()

    def forward(self, node_emb, relation):
        r_emb = self.rela_emb[relation]
        while len(r_emb.size()) != len(node_emb.size()):
            r_emb = r_emb.unsqueeze(1)
        return node_emb - (node_emb * r_emb).sum(-1, keepdim=True) * r_emb

    def constraint(self):
        self.rela_emb.data = F.normalize(self.rela_emb.data)


class RelaDistMult(nn.Module):

    def __init__(self, n_relation, emb_size, normalize=True, **kwargs):
        super().__init__()
        # self.rela_emb = nn.Embedding(n_relation, emb_size)
        self.rela_emb = nn.Parameter(torch.zeros([n_relation, emb_size]))
        self.emb_size = emb_size
        self.normalize = normalize
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.normalize:
            nn.init.constant_(self.rela_emb, 1)
        else:
            nn.init.xavier_uniform_(self.rela_emb)
        self.constraint()

    def forward(self, node_emb, relation):
        r_emb = self.rela_emb[relation]
        while len(r_emb.size()) != len(node_emb.size()):
            r_emb = r_emb.unsqueeze(1)
        if self.normalize:
            return node_emb * r_emb * math.sqrt(self.emb_size)
        else:
            return node_emb * torch.tanh(r_emb)

    def constraint(self):
        if self.normalize:
            self.rela_emb.data = F.normalize(self.rela_emb.data)


class RelaBinomial(nn.Module):

    def __init__(self, n_relation, emb_size, upscale=False, **kwargs):
        super().__init__()
        self.rela_emb_ = nn.Parameter(torch.zeros([n_relation, emb_size]))
        self.emb_size = emb_size
        self.upscale = upscale
        self.reset_parameters()
    
    @property
    def rela_emb(self):
        return torch.sigmoid(self.rela_emb_)
    
    def reset_parameters(self):
        nn.init.normal_(self.rela_emb_)

    def forward(self, node_emb, relation):
        rela_emb = self.rela_emb
        if self.upscale:
            rela_emb = rela_emb * self.emb_size / rela_emb.sum(-1, keepdim=True)
            rela_emb = torch.clamp(rela_emb, 0, 3)
        r_emb = rela_emb[relation]
        while len(r_emb.size()) != len(node_emb.size()):
            r_emb = r_emb.unsqueeze(1)
        return node_emb * r_emb


class DistMult(nn.Module):

    def __init__(self, emb_size, **kwargs):
        super(DistMult, self).__init__()
        self.rela_emb = nn.Parameter(torch.ones([emb_size]))
        nn.init.normal_(self.rela_emb.data)

    def forward(self, node_emb):
        return node_emb * self.rela_emb

class Linear(nn.Linear):
    
    def __init__(self, emb_size, bias=False, **kwargs):
        super().__init__(emb_size, emb_size, bias=bias)


class Binomial(nn.Module):

    def __init__(self, emb_size, upscale=False, **kwargs):
        super().__init__()
        self.rela_emb_ = nn.Parameter(torch.ones([emb_size]))
        nn.init.normal_(self.rela_emb_.data)
        self.upscale = upscale

    @property
    def rela_emb(self):
        return torch.sigmoid(self.rela_emb_)
    
    def forward(self, node_emb):
        v = torch.sigmoid(self.rela_emb_)
        if self.upscale:
            v = v * v.size(0) / v.sum()
            v = torch.clamp(v, 0, 3)
        return v * node_emb


class Block(nn.Module):

    def __init__(self, emb_size, conv_size=16, **kwargs):
        super().__init__()
        assert emb_size % conv_size == 0
        self.emb_size = emb_size
        self.conv_size = conv_size
        self.num_block = emb_size // conv_size
        self.blocks = nn.Parameter(torch.zeros(self.num_block, self.conv_size, self.conv_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        xavier_uniform_(self.blocks, self.conv_size, self.conv_size)
    
    def forward(self, node_emb):
        conv = block_diag(self.blocks)
        return F.linear(node_emb, conv)


class TransH(nn.Module):

    def __init__(self, emb_size, **kwargs):
        super().__init__()
        self.rela_emb = nn.Parameter(torch.ones([emb_size, ]))
        self.constraint()

    def forward(self, node_emb):
        return node_emb - (node_emb * self.rela_emb).sum(-1, keepdim=True) * self.rela_emb

    def constraint(self):
        self.rela_emb.data = F.normalize(self.rela_emb.data, dim=0)


class RelationTransform(nn.Module):

    def __init__(self, n_relation, emb_size, transform, input_size=None, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.n_relation = n_relation
        if input_size is not None and input_size != emb_size:
            self.rela_fn = nn.ModuleList([
                nn.Linear(input_size, emb_size, bias=False) for _ in range(n_relation)
            ])
            if transform != 'linear':
                print('only support linear transformation when input_size != emb_size')
        else:
            self.rela_fn = nn.ModuleList([get_transform(transform)(emb_size) for _ in range(n_relation)])

    def __len__(self):
        return len(self.rela_fn)

    def __iter__(self):
        return iter(self.rela_fn)

    def forward(self, node_emb, relation):
        out = torch.zeros([*node_emb.shape[:-1], self.emb_size], device=node_emb.device)
        for r in range(self.n_relation):
            mask = relation == r
            emb = node_emb[mask]
            if emb.size(0) == 0:
                continue
            out[mask] = self.rela_fn[r](emb)
        return out
