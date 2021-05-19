import torch
import torch.nn as nn
import torch.nn.functional as F
from model.scoring import Matcher


class Discriminator(nn.Module):

    def __init__(self, ctx_dim, loc_dim, use_norm=True, bias=False, loss_func='bce'):
        super().__init__()
        assert loss_func in {'bce', 'ce'}
        self.disc = nn.Linear(ctx_dim, loc_dim, bias=False)
        self.matcher = Matcher(use_norm=use_norm)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = 0
        self.loss_func = loss_func

    def forward(self, ctx_pos, ctx_neg, loc_pos, loc_neg):
        ctx_pos = self.disc(ctx_pos)
        ctx_neg = self.disc(ctx_neg)
        sc_pos = self.matcher(ctx_pos, loc_pos) + self.bias
        sc_neg = self.matcher(ctx_neg, loc_neg) + self.bias
        if self.loss_func == 'bce':
            return (F.binary_cross_entropy_with_logits(sc_pos, torch.ones_like(sc_pos), reduction='none') + \
            F.binary_cross_entropy_with_logits(
                sc_neg, torch.zeros_like(sc_neg), reduction='none')) / 2
        else:
            target = torch.zeros_like(sc_pos).long().squeeze(1)
            sc = torch.cat([sc_pos, sc_neg], dim=1)
            return F.cross_entropy(sc, target, reduction='none')

