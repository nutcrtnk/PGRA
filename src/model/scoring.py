import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from model.module.transform import get_relation_transform
from torch_scatter import scatter
import numpy as np

class Matcher(nn.Module):

    def __init__(self, mode='dot', use_norm=False):
        super().__init__()
        self.mode = mode
        self.use_norm = use_norm
    
    def forward(self, tx, ty, r=None):
        if self.use_norm:
            tx = tx / (torch.norm(tx, dim=-1, keepdim=True) + 1e-12)
            ty = ty / (torch.norm(tx, dim=-1, keepdim=True) + 1e-12)
        if self.mode == 'dot':
            return tx * ty
        elif self.mode == 'l1':
            return -torch.abs(tx-ty)
        elif self.mode == 'l2':
            return -(tx-ty) ** 2
        else:
            raise Exception('invalid scoring mode')

    def __repr__(self):
        return '{}({}, norm={})'.format(
            self.__class__.__name__, self.mode, self.use_norm)

def weight_loss(loss_func):
    def weight_loss_func(*args, weights=None, reduce='mean', **kwargs):
        loss = loss_func(*args, **kwargs)
        if weights is not None:
            while len(weights.size()) != len(loss.size()):
                weights = weights.unsqueeze(-1)
            loss *= weights
        if reduce is None or reduce == 'none':
            return loss
        elif reduce == 'mean':
            return loss.mean()
        elif reduce == 'sum':
            return loss.sum()
        else:
            raise Exception('invalid reduce')
    return weight_loss_func

@weight_loss
def bce_loss(p_score, n_score, smoothing=0.):
    labels = torch.cat([torch.ones_like(p_score), torch.zeros_like(n_score)], dim=-1)
    labels = ((1.0-smoothing)*labels) + smoothing * (1-labels)
    scores = torch.cat([p_score, n_score], dim=-1)
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    return loss

@weight_loss
def ce_loss(p_score, n_score):
    target = torch.zeros_like(p_score).long().flatten()
    scores = torch.cat([p_score, n_score], dim=-1)
    return F.cross_entropy(scores, target, reduction='mean')

@weight_loss
def bpr_loss(p_score, n_score):
    diff = p_score - n_score
    loss = -F.logsigmoid(diff)
    return loss

@weight_loss
def margin_loss(p_score, n_score, margin=0.):
    loss = F.relu(n_score - p_score + margin)
    return loss

@weight_loss
def softplus_loss(p_score, n_score):
    loss_func = lambda x, y: torch.mean(F.softplus(-y * x), dim=-1)
    loss = loss_func(p_score, 1) / 2
    loss += loss_func(n_score, -1) / 2
    return loss

def get_loss_function(loss_name):
    _losses = {
        'bce': bce_loss,
        'bpr': bpr_loss,
        'softplus': softplus_loss,
        'ce': ce_loss,
    }
    if loss_name in _losses:
        return _losses[loss_name]
    if loss_name[:6] == 'margin':
        return partial(margin_loss, margin=float(loss_name[6:]))
    if loss_name[:3] == 'bce':
        return partial(bce_loss, smoothing=float(loss_name[3:]))
    raise ValueError('Invalid loss_name')


class Monitor:

    def __init__(self, n_relation, loss):
        self.count = torch.zeros([n_relation])
        self.losses = torch.zeros([n_relation])
        self.loss_func = get_loss_function(loss)
        self.n_relation =  n_relation

    def summarize(self):
        return self.losses / (self.count + 1e-12)

    def reset(self):
        self.count = torch.zeros_like(self.count)
        self.losses = torch.zeros_like(self.losses)

    def update(self, p_score, n_score, relation):
        with torch.no_grad():
            losses = self.loss_func(p_score, n_score, reduce='none')
            losses = losses.view(losses.size(0), -1).mean(dim=-1)
            idx = torch.argsort(relation)
            losses, relation = losses[idx], relation[idx]
            self.count += scatter(torch.ones_like(relation), relation, dim_size=self.n_relation).cpu()
            self.losses += scatter(losses, relation, dim_size=self.n_relation).cpu()


class Scoring(torch.nn.Module):

    def __init__(self, use_norm=False, score='dot', loss='bpr', temp=1, rel='none', emb_size=None, n_relation=None, rel_weight=None, **kwargs):
        super().__init__()
        self.loss = loss
        self.rel_transform = get_relation_transform(rel)(n_relation, emb_size)
        self.temp = temp
        self.score_func = Matcher(use_norm=use_norm, mode=score)
        self.loss_func = get_loss_function(loss)
        self.monitor = Monitor(n_relation, loss)

        if rel_weight is not None:
            if not isinstance(rel_weight, torch.Tensor):
                rel_weight = torch.from_numpy(np.array(rel_weight))
            self.register_buffer('rel_weight', rel_weight)
        else:
            self.rel_weight = None

    def forward(self, p1_feat, p2_feat, n1_feat, n2_feat, relation):
        p1_feat = self.rel_transform(p1_feat, relation)
        p1_feat = p1_feat.unsqueeze(1)
        p2_feat = p2_feat.unsqueeze(1)
        p_score = self.score_func(p1_feat, p2_feat) / self.temp
        n2_score = self.score_func(p1_feat.expand_as(n2_feat), n2_feat) / self.temp
        n1_feat = self.rel_transform(n1_feat, relation)
        n1_score = self.score_func(n1_feat, p2_feat.expand_as(n1_feat)) / self.temp
        p_score, n1_score, n2_score = p_score.sum(-1), n1_score.sum(-1), n2_score.sum(-1)
        self.monitor.update(p_score, n1_score, relation)
        self.monitor.update(p_score, n2_score, relation)
        weights = self.rel_weight[relation] if self.rel_weight is not None else None
        loss = self.loss_func(p_score, n2_score, weights=weights)
        loss += self.loss_func(p_score, n1_score, weights=weights)
        return loss

    def predict(self, p1_feat, p2_feat, relation):
        p1_feat = self.rel_transform(p1_feat, relation)
        p1_feat = p1_feat.unsqueeze(1) if len(p1_feat.size()) == 2 else p1_feat
        p2_feat = p2_feat.unsqueeze(1) if len(p2_feat.size()) == 2 else p2_feat
        score = self.score_func(p1_feat, p2_feat).sum(dim=-1) / self.temp
        return score
