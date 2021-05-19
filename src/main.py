from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
import ast
import random
import sys
import numpy as np
import os
import torch

if not os.environ.get("RAND", False):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
else:
    print('random seed')


import config
from pathlib import Path
from train import Trainer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def none_or_str(s):
    if s.lower() == 'none':
        return None
    return s
    
def none_or_value(s):
    if s.lower() == 'none':
        return None
    return ast.literal_eval(s)

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('dataset', help='Input graph file')
    parser.add_argument('--task', type=str, default='link')
    parser.add_argument('--out', help='Output representation file', default=None)
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--emb_size', default=128, type=int)

    parser.add_argument('--pre', type=str, default='distmult')
    parser.add_argument('--score', type=str, default='dot')
    parser.add_argument('--degree', default=2, type=int)
    parser.add_argument('--nb', default=20, type=int)
    parser.add_argument('--self_loop', default=1, type=float)

    parser.add_argument('--n_neg', default=5, type=int)
    parser.add_argument('--test_neg', default=10, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--max_epochs', default=None, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--rel', default='none', type=str)

    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--loss', default='bpr', type=str)
    # parser.add_argument('--norm', default=True, type=str2bool)
    parser.add_argument('--norm', default=0, type=float)
    parser.add_argument('--nb_reg', default=0.001, type=float)
    parser.add_argument('--best_lambda', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-p', '--patience', default=10, type=int)
    parser.add_argument('--lr_patience', default=2, type=int)
    parser.add_argument('--load', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--metric', type=str, default='mrr') # step, loss, {metric}
    parser.add_argument('--eval_step', type=int, default=None)
    parser.add_argument('--val', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['mrr'])
    parser.add_argument('--large', type=str2bool, nargs='?', const=True, default=False)
    args, unknown = parser.parse_known_args()
    return args, unknown


def main(args, unknown=None):
    if args.out is None:
        args.out = f'PGRA_{args.dataset}'
    if args.task == 'link':
        assert args.metric in ('mrr', 'mrr_micro', 'mrr_macro')

    if args.best_lambda:
        from model.best_lambda import get_lambda
        ds, n_test, *_ = args.dataset.split('_')
        n_test = float(n_test)
        args.nb_reg = get_lambda(args.score, ds, n_test) # / (args.degree * (args.degree + 1) / 2)
        print('change nb_reg to:', args.nb_reg)

    added_kwargs = {}
    if unknown is not None:
        def _parse(value):
            try:
                return ast.literal_eval(value)
            except:
                return value
        
        print(unknown)
        for k, v in zip(unknown[::2], unknown[1::2]):
            if k[:2] != '--':
                raise Exception('need -- for kwargs')
            k = k[2:].replace('-', '_')
            v = v.split(',')
            if len(v) == 1:
                v = _parse(v[0])
            else:
                v = [_parse(_v) for _v in v]
            added_kwargs[k] = v
    
    t1 = time.time()
    print("Reading...")

    if args.load:
        trainer = Trainer.load(args.dataset, args.out)
        if args.val:
            print('Val:', trainer.eval(large=args.large, metrics=args.eval_metrics))
        if args.test:
            print('Test:', trainer.eval(val=False, large=args.large, metrics=args.eval_metrics))
    else:
        kwargs = dict(out=args.out, task=args.task, emb_size=args.emb_size, batch_size=args.batch_size, n_neighbor=args.nb, loss=args.loss, degree=args.degree, n_neg=args.n_neg, temp=args.temp, norm=args.norm, rel=args.rel, self_loop=args.self_loop, pre=args.pre, score=args.score, nb_reg=args.nb_reg)
        kwargs.update(added_kwargs)
        trainer = Trainer(args.dataset, **kwargs)
    
        trainer.run(lr=args.lr, weight_decay=args.decay, patience=args.patience, max_steps=args.max_steps, max_epochs=args.max_epochs, metric=args.metric, eval_step=args.eval_step, lr_patience=args.lr_patience)
        print('total time:', time.time() - t1)

    if args.dataset in ('amazon', 'youtube', 'twitter'):
        from data.hin_reader import __read_hin_splited
        path = config.hin_dir / args.dataset
        edges = __read_hin_splited(path / 'test.txt', 'r h t w')
        from collections import defaultdict
        true_edges = defaultdict(list)
        false_edges = defaultdict(list)
        g = trainer.graph
        for d in edges:
            if args.dataset == 'twitter' and d['r'] != '1':
                continue
            h = g.node2id(d['h'], d['a'])
            t = g.node2id(d['t'], d['b'])
            r = g.et2id(d['r'])
            e = (h, t)
            if d['w'] == 0:
                false_edges[r].append(e)
            else:
                true_edges[r].append(e)
        scores = trainer.eval_f1auc(true_edges, false_edges)
        auc, pr, f1 = scores['auc'], scores['pr'], scores['f1']
        print('Overall ROC-AUC:', auc)
        print('Overall PR-AUC:', pr)
        print('Overall F1:', f1)
    return trainer


if __name__ == "__main__":
    main(*parse_args())

