import numpy as np
import torch
import copy
from functools import partial

class LossTracker:

    def __init__(self):
        self.step = 0
        self.losses = None

    def update(self, losses):
        if not isinstance(losses, tuple) and not isinstance(losses, list):
            losses = (losses,)
        losses = np.array([x.item() for x in losses])
        if self.losses is None:
            self.losses = losses
        else:
            self.losses = ((self.losses * self.step) + losses) / (self.step + 1)
        self.step += 1

    @property
    def value(self):
        return sum(self.losses)

    def __str__(self):
        return "{:.5f} ".format(sum(self.losses)) + ' '.join("{:.5f}".format(loss) for loss in self.losses)


class MultiClsTracker:

    def __init__(self, metrics, n_classes):
        self.metric_scores = {metric: np.zeros(n_classes) for metric in metrics}
        self.n_samples = np.zeros(n_classes)
        self.num_classes = n_classes
        self.metric_summary = {}
        self._is_updated = False
        self.metric_func = {metric: get_metric(metric) for metric in metrics}
        self.edges_scores = {metric: [] for metric in metrics}
        self.same_score = 0

    def __len__(self):
        return int(self.n_samples.sum())

    def update(self, pred_scores, classes, positive_arg=0, edges=None):
        argsort = torch.argsort(pred_scores, dim=1, descending=True)
        for i in range(len(pred_scores)):
            pos = positive_arg if isinstance(positive_arg, int) else positive_arg[i]
            same = (pred_scores[i, :] == pred_scores[i, pos]).sum().item()
            self.same_score += same - 1
            ranking = (argsort[i, :] == pos).nonzero()
            ranking = 1 + ranking.item()
            c = classes if isinstance(classes, int) else classes[i]
            for metric, values in self.metric_scores.items():
                metric_score = np.mean([self.metric_func[metric](ranking + j) for j in range(same)])
                values[c] += metric_score
                if edges is not None:
                    self.edges_scores[metric].append([edges[i], metric_score])
            self.n_samples[c] += 1
        self._is_updated = True

    def summarize(self):
        metric_summary = {}
        mask = self.n_samples > 0
        n_samples = self.n_samples[mask]
        for metric, values in self.metric_scores.items():
            values = values[mask]
            metric_summary[metric + '_macro'] = (values / n_samples).mean()
            metric_summary[metric + '_micro'] = values.sum() / n_samples.sum()
            metric_summary[metric] = (metric_summary[metric + '_micro'] + metric_summary[metric + '_macro']) / 2
        self.metric_summary = metric_summary
        self._is_updated = False
        return metric_summary

    def __getitem__(self, item):
        if self._is_updated:
            self.summarize()
        return self.metric_summary[item]

    def value(self):
        return copy.deepcopy(self.metric_summary)

    @property
    def macro(self):
        mask = self.n_samples > 0
        return {metric: values[mask] / self.n_samples[mask] for metric, values in self.metric_scores.items()}

    def __str__(self):
        if self._is_updated:
            self.summarize()
        return ', '.join([metric + ": {:.5f}".format(score) for metric, score in self.metric_summary.items()])

    def __format__(self, format_spec):
        return str(self)


def get_metric(metric):
    metric = metric.lower()
    if metric[:5] == 'hits@':
        return partial(hits, k=int(metric[5:]))
    if metric[:5] == 'ndcg@':
        return partial(ndcg, k=int(metric[5:]))
    mapper = {
        'mrr': mrr,
        'mr': mr,
        'ndcg': ndcg,
    }
    return mapper[metric]

def ndcg(ranking, k=0):
    if k <= 0 or ranking <= k:
        return 1 / np.log2(1+ranking)
    else:
        return 0

def mrr(ranking):
    return 1.0 / ranking

def mr(ranking):
    return float(ranking)

def hits(ranking, k):
    return 1.0 if ranking <= k else 0.
