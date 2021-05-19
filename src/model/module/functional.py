import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def partialclass(cls, new_name, *args, **kwds):
    class NewCls(cls):
        def __init__(self, *cls_args, **cls_kwds):
            cls_args = args + tuple(cls_args)
            cls_kwds.update(kwds)
            super().__init__(*cls_args, **cls_kwds)
    NewCls.__name__ = new_name
    return NewCls

def get_activation(act):
    if act is None or act in ['iden', 'none']:
        return lambda x: x
    act = act.lower()
    if hasattr(F, act):
        return getattr(F, act)
    elif act == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise Exception('No activation named:', act)

def soft_normalize(tensor, norm=1):
    node_emb_norm = torch.norm(tensor, dim=-1, keepdim=True)
    return torch.where(node_emb_norm > norm, F.normalize(tensor, dim=-1) * norm, tensor)

def normalize(tensor, norm=1):
    return F.normalize(tensor, dim=-1) * norm

def xavier_uniform_(tensor, fan_in, fan_out, gain=1):    
    std = math.sqrt(2.0 / (fan_in + fan_out))
    a = gain * math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-a, a)

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))