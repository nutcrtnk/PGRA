import os
import torch
import torch.nn as nn
import torch.nn.functional as F

if not os.environ.get("RAND", False):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print('random seed')

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes, rela=1, drop=0.):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in // rela, nb_classes)
        self.rela = rela
        if rela > 1:
            self.att = nn.Parameter(torch.zeros([rela, 1]))
            torch.nn.init.xavier_uniform_(self.att)
        else:
            self.att = None
        for m in self.modules():
            self.weights_init(m)
        self.dropout = nn.Dropout(drop)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, seq):
        seq = self.dropout(seq)
        if self.att is not None:
            att = F.softmax(self.att, dim=0)
            seq = (seq.view(seq.shape[0], self.rela, -1) * att).sum(1)
        ret = self.fc(seq)
        return ret
