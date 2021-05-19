from model.gnn import GNN
from model.scoring import Scoring
from data.link_gen import LinkGenerator, init_fn
from torch.utils.data import DataLoader
from time import perf_counter, time
import torch.nn.functional as F
import torch
import numpy as np
import config
import tempfile
from collections import Counter
from tqdm import tqdm
import os
from model.evaluator import NodeEvaluator
from tracker import LossTracker, MultiClsTracker
import ast
from data.label import EdgeLabel
from data.hin_reader import read_hin_splited, read_hin, read_dmgi
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
import re
import copy
from inspect import signature

def model_config(func):
    def new_func(*args, **kwargs):
        init_sig = signature(func).bind(*args, **kwargs)
        init_sig.apply_defaults()
        params = init_sig.arguments
        del params['self']
        m_config = {}
        for key, value in params.items():
            if re.fullmatch('<.* object .*>', str(value)) is None:
                m_config[key] = copy.deepcopy(value)
        func(*args, **kwargs)
        if hasattr(args[0], 'config'):
            args[0].config.update(m_config)
        else:
            args[0].config = m_config
    return new_func


class Trainer:

    _default_optim = 'Adam'
    _default_optim_params = {'lr': 5e-4}

    @model_config
    def __init__(self, dataset, out, task='link', no_feature=False, emb_size=128, degree=2, n_neighbor=20, batch_size=100, n_neg=5, num_workers=4, score='dot', loss='bpr', row_norm=True, use_node_type=True, self_loop=1, test_batch_size=None, temp=1, rel='none', matcher_norm=False, balance=False, **kwargs):
        super().__init__()
        assert task in ('link', 'node')
        self.task = task
        self.out = out
        self.dataset_name = dataset

        if test_batch_size is None:
            test_batch_size = batch_size // 4

        self.data_loader_task = None

        if self.task == 'link':
            graph, valid_data, test_data = read_hin_splited(dataset)
            self.data_loader_val = DataLoader(
                valid_data, batch_size=test_batch_size, collate_fn=LinkGenerator.collate_fn, shuffle=True)
            self.data_loader_test = DataLoader(
                test_data, batch_size=test_batch_size, collate_fn=LinkGenerator.collate_fn)

        else:
            if dataset[:3] == 'acm' or dataset[:4] == 'imdb' or dataset.startswith('attr_'):
                graph = read_dmgi(dataset)
            else:
                graph = read_hin(dataset)
            self.data_loader_val, self.data_loader_test = None, None
        self.graph = graph
        self.corrupted_nodes = None
        self.corrupted_features = None
        self.edge_label = EdgeLabel(self.graph)
        self.n_edge_types = graph.edge_type_size

        if balance:
            edge_type_count = Counter(self.edge_label.labels)
            edge_type_count = np.array([edge_type_count[i] for i in range(self.n_edge_types)])
            rel_weight = 1./edge_type_count / (sum(1. / edge_type_count)) * self.n_edge_types
            print('rel_weight:', rel_weight)
        else:
            rel_weight = None            

        self.emb_size = emb_size
        self._cuda = False
        
        if self.task == 'link':
            subG = self.graph.G.edge_subgraph(self.edge_label.train_edges)
        else:
            subG = self.graph.G

        self.batch_size = batch_size
        self.self_loop = self_loop
        self.subG = subG
        if isinstance(n_neighbor, int):
            self.n_neighbor = n_neighbor
        else:
            self.n_neighbor = max(n_neighbor)

        self.scoring = Scoring(score=score, use_norm=matcher_norm, loss=loss, rel=rel, emb_size=emb_size, n_relation=self.n_edge_types, temp=temp, rel_weight=rel_weight)

        if not graph.use_one_hot and not no_feature:
            features = graph.get_nodes_features()
            features = torch.Tensor(features).float()
            if row_norm:
                features = features / (features.sum(-1, keepdim=True) + 1e-12)
        else:
            features = None
        if features is not None:
            print('use features')

        adj_node, adj_rela = self.create_node_neighbors()

        node_type = torch.zeros([graph.node_size]).long()
        nodes_of_types = graph.nodes_of_types
        for i, t in enumerate(sorted(nodes_of_types)):
            node_type[nodes_of_types[t]] = i

        self.model = GNN(graph.node_size, self.n_edge_types, emb_size, features=features, 
                          n_hop=degree, n_neighbor=adj_node.shape[1], node_type=node_type, self_loop=self_loop, **kwargs)
        self.model.register_neighbors(adj_node, adj_rela)
        self.model.reset()
        self.nodes = self.subG.nodes

        nodes_of_types = graph.nodes_of_types if use_node_type else None

        link_gen_train = LinkGenerator(self.edge_label, graph.nodes_of_types, n_neg=n_neg, use_type=use_node_type)
        print('Number of training edges:', len(link_gen_train))
        self.data_loader_train = DataLoader(link_gen_train, batch_size=batch_size, num_workers=num_workers,
                                            shuffle=True, collate_fn=LinkGenerator.collate_fn, worker_init_fn=init_fn)
        self.data_loader_train_iter = None
        if test_batch_size is None:
            test_batch_size = batch_size // 4
        if self.task == 'node':
            self.node_eval = NodeEvaluator(self.graph)
        else:
            self.node_eval = None
        self.used_keys = ['r', 'h', 't', 'h_neg', 't_neg']

        self.optim = self._default_optim
        self.optim_params = self._default_optim_params
        self.optimizer = None
        self.total_epoch = 0
        self.total_steps = 0
        self.degree = degree

        self.temp_dir = tempfile.mkstemp()[1]
        self.best_iter = 0
        self.best_scores = None
        self.test_scores = None
        self.cooldown = 0
        self.history = []
        self.device = torch.device("cuda:0")
        
        self.total_training_time = 0

    def create_node_neighbors(self):
        print('create node neighbors')
        adj_entity = np.ones(
            [self.graph.node_size, self.n_neighbor], dtype=np.int64) * -1
        adj_relation = np.ones(
            [self.graph.node_size, self.n_neighbor], dtype=np.int64) * -1
        if self.self_loop:
            adj_entity[:, 0] = np.arange(adj_entity.shape[0])
            adj_relation[:, 0] = self.n_edge_types
        for node, node_type in self.subG.nodes(data='node_type'):
            nbs = list(self.subG.edges(node, data='label'))
            num_sampled = 0 if not self.self_loop else 1
            sample_pool = set(range(len(nbs)))

            sampled = list(sample_pool)
            if len(sampled) + num_sampled > self.n_neighbor:
                sampled = np.random.choice(
                    sampled, size=self.n_neighbor - num_sampled, replace=False)

            adj_entity[node, num_sampled:num_sampled +
                       len(sampled)] = [nbs[x][1] for x in sampled]
            adj_relation[node, num_sampled:num_sampled +
                         len(sampled)] = [nbs[x][2] for x in sampled]
        return adj_entity, adj_relation

    def run(self, lr=5e-4, weight_decay=0, cuda=True, optim='Adam', **kwargs):
        if cuda:
            self.cuda()
        self.create_optimizer(
            optim, lr=lr, weight_decay=weight_decay)
        self.train(**kwargs)

    def train(self, max_steps=50000, max_epochs=None, patience=10, metric='mrr', save=True, eval_step=None, lr_patience=2):
        assert self.optimizer is not None
        t0 = perf_counter()
        if eval_step is None:
            eval_step = len(self.data_loader_train)
            if eval_step > 500:
                eval_step = 500
        elif eval_step <= 0:
            eval_step = len(self.data_loader_train)                
        patience *= eval_step
        self._save(self.temp_dir)
        print()
        print('===== Training =====')

        def check_best(scores):
            if self.best_scores is None:
                return True
            if isinstance(scores, dict):
                return scores[metric] > self.best_scores[metric]
            return scores > self.best_scores

        def update_best(scores):
            nonlocal is_best, best_step
            if check_best(scores):
                self.best_scores, best_step = scores, step
                self.best_iter = best_step
                is_best = True
                print('update best')

        self.best_scores = None
        best_step = 0
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=lr_patience, verbose=True, cooldown=20, threshold=1e-3)

        for step in range(0, max_steps, eval_step):

            t = perf_counter()
            train_losses = self.train_one_epoch(steps=eval_step)
            self.total_training_time += perf_counter() - t
            print('Rel losses:', self.scoring.monitor.summarize())
            self.scoring.monitor.reset()

            is_best = False
            scores = None
            if metric != 'loss' and self.data_loader_val is not None:
                scores = self.eval(mode='val')
                print("Epoch:", '%04d' % (self.total_epoch + 1), "Step:", step, "train_loss=", train_losses,
                      "Val:", scores, "time=", "{:.5f}".format(perf_counter()-t), flush=True)
                scores = scores.metric_summary
                update_best(scores)
                if os.environ.get("TEST", False):
                    _scores = self.eval(mode='test')
                    print('Dist epochs:', step - best_step)
                    print("Test:", _scores)
            else:
                print("Epoch:", '%04d' % (self.total_epoch + 1),
                      "Step:", step, "train_loss=", train_losses, "time=", "{:.5f}".format(perf_counter()-t))
            
            if self.node_eval is not None:
                val_s, _ = self.eval_node(n_runs=5)
                val_s = sorted(val_s.values(), key=lambda x: x[metric])[-1]
                update_best(val_s)

            self.history.append(
                (self.total_epoch+1, step, train_losses.value, scores))

            if is_best and save:
                self._save(self.temp_dir)
            if step - best_step > patience and self.cooldown == 0:
                print("Early stopping...")
                break

            if max_epochs and self.total_epoch + 1 == max_epochs:
                print('Reach max epochs...')
                break

            self.cooldown -= self.cooldown > 0
            reduce_lr.step(train_losses.value)
            if reduce_lr.cooldown_counter == reduce_lr.cooldown:
                reduce_lr.best = reduce_lr.mode_worse

        train_time = perf_counter() - t0
        self._load(self.temp_dir)
        print("Optimization Finished!")
        print("Actual Training time: {:.4f}s".format(self.total_training_time,))
        print("Train time: {:.4f}s".format(train_time,))
        if save:
            print('Save model')
            self.save()

        if self.data_loader_val is not None:
            val_scores = self.eval(mode='val')
            print("Val:", val_scores)

        if self.data_loader_test is not None:
            self.test_scores = self.eval(mode='test')
            print("Test:", self.test_scores)

        if self.node_eval is not None:
            self.eval_node()

    def eval_node(self, cls_test=True, n_runs=50):
        print(f'Test: ({n_runs})')
        n_rela = self.n_edge_types
        with torch.no_grad():
            self.model.eval()
            emb_list = [self.get_all_rela_embedding(
                r, cpu=False) for r in range(n_rela)]
        val_rel_scores, test_rel_scores = {}, {}
        for i, _emb in enumerate(emb_list):
            edge_type = None
            for k, v in self.graph.edge_type_to_id.items():
                if v == i:
                    print('EdgeType:', k)
                    edge_type = k
            if edge_type is not None:
                val_scores, test_scores = self.node_eval.run(_emb, cls_test=cls_test, n_runs=n_runs)
                test_rel_scores[edge_type] = test_scores
                val_rel_scores[edge_type] = val_scores
            else:
                print('invalid', i, len(emb_list))
        return val_rel_scores, test_rel_scores

    def cuda(self):
        self._cuda = True
        self.model.to(self.device)
        self.scoring.to(self.device)
        if self.node_eval is not None:
            self.node_eval.cuda()

    def create_optimizer(self, optim=None, **kwargs):
        if optim is not None:
            self.optim = optim
        self.optim_params.update(kwargs)
        params = list(self.model.parameters()) + list(self.scoring.parameters())
        self.optimizer = getattr(torch.optim, self.optim)(
            params, **self.optim_params)

    def save(self):
        self._save(self._get_model_path() / 'model.pt')
        self._save_config(self._get_model_path() / 'config.txt')

    def _save(self, path):
        state = {
            'model': self.model.state_dict(),
            'scoring': self.scoring.state_dict(),
            'best_iter': self.best_iter
        }
        torch.save(state, path)

    def _save_config(self, path):
        with open(path, 'w') as f:
            f.write('{}'.format(self.config))

    @staticmethod
    def load(dataset, name, strict=True, **kwargs):
        path = Trainer.get_model_path(dataset, name, build_path=False)
        with open(path / 'config.txt') as f:
            cfg = ast.literal_eval(f.read())
            if 'kwargs' in cfg:
                cfg.update(cfg['kwargs'])
                del cfg['kwargs']
            cfg['dataset'] = dataset
            cfg.update(kwargs)
            print(cfg)
        trainer = Trainer(**cfg)
        trainer._load(path / 'model.pt', strict=strict)
        return trainer

    def _load(self, path, strict=True):
        state = torch.load(str(path), map_location='cpu')
        self.model.load_state_dict(state['model'], strict=strict)
        if 'scoring' in state:
            self.scoring.load_state_dict(state['scoring'])
        self.best_iter = state['best_iter']
        if self._cuda:
            self.cuda()

    def _get_model_path(self, build_path=True):
        return self.get_model_path(self.dataset_name, self.out, build_path=build_path)

    @staticmethod
    def get_model_path(dataset, name, build_path=True):
        folder = config.model_dir / dataset / name
        if build_path:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def _join(*args, sep='_'):
        return sep.join([str(arg) for arg in args])

    def get_train_data(self):
        if self.data_loader_train_iter is None:
            self.data_loader_train_iter = iter(self.data_loader_train)
        try:
            data = next(self.data_loader_train_iter)
        except StopIteration:
            self.data_loader_train_iter = iter(self.data_loader_train)
            self.total_epoch += 1
            data = next(self.data_loader_train_iter)
        return data

    def train_one_epoch(self, steps=None):
        if steps is None:
            steps = len(self.data_loader_train)
        pbar = tqdm(range(steps), disable=not os.environ.get("TQDM", False))
        losses = LossTracker()
        self.model.train()
        t = 0

        def update(loss):
            loss.backward()
            self.optimizer.step()
            for m in self.model.modules():
                if hasattr(m, 'inner_loss'):
                    m.inner_loss = torch.zeros_like(m.inner_loss)
            
            self.optimizer.zero_grad()
            self.model.constraint()

        for step in pbar:

            _t = time()
            train_losses = []
            reg_losses = []
            all_data = self.get_train_data()
            all_data = {k: v.to(self.device) for k, v in all_data.items()}

            data = [all_data[x] for x in self.used_keys]
            r = data[0]
            nodes_vec = [self.get_embedding(x, r)
                         for i, x in enumerate(data[1:])]
            link_losses = self.scoring(*nodes_vec, relation=r)
            if not isinstance(link_losses, tuple) and not isinstance(link_losses, list):
                link_losses = [link_losses]
            train_losses += link_losses
            reg_losses += [m.inner_loss for m in self.model.modules()
                           if hasattr(m, 'inner_loss') and m.inner_loss.item() > 0]
            train_losses = train_losses + reg_losses

            update(sum(train_losses))
            losses.update(train_losses)
            t += time() - _t
            pbar.set_description('epoch {} loss:{} time:{:.2f}'.format(
                self.total_epoch, losses, t*len(pbar)/(step+1)))

            self.total_steps += 1

        return losses

    def eval(self, mode='val', metrics=('mrr',), large=False):
        data_loader = getattr(self, 'data_loader_{}'.format(mode))
        _eval = self._eval_large_neg if large else self._eval
        return _eval(data_loader, metrics, n=60000 if mode == 'val' else -1)

    def _eval(self, data_loader, metrics, n=-1):
        self.model.eval()
        lp_tracker = MultiClsTracker(metrics, self.n_edge_types)
        with torch.no_grad():
            for num, all_data in enumerate(data_loader):
                data = [all_data[x] for x in self.used_keys]
                if self._cuda:
                    data = [x.to(self.device) for x in data]
                r, p1, p2, n1, n2 = data
                pos, neg = (p1, p2), (n1, n2)
                for i in (0, 1):
                    feat1 = self.get_embedding(
                        torch.cat((pos[i].unsqueeze(-1), neg[i]), dim=-1), r, is_head=i)
                    feat2 = self.get_embedding(pos[1-i], r, is_head=i)
                    feat_args = (feat1, feat2) if i == 0 else (feat2, feat1)
                    ratings = self.scoring.predict(*feat_args, relation=r)
                    lp_tracker.update(ratings, r, edges=torch.stack(
                        [p1, p2, r], dim=-1).cpu().numpy())
                if 0 < n < len(lp_tracker):
                    break
        for metric in lp_tracker.macro:
            print({metric: lp_tracker.macro[metric]})
        lp_tracker.summarize()
        return lp_tracker

    def _eval_large_neg(self, data_loader, metrics, n=-1):
        self.model.eval()
        lp_tracker = MultiClsTracker(metrics, self.n_edge_types)
        with torch.no_grad():
            for _edge in range(self.n_edge_types):
                embs = self.get_all_rela_embedding(_edge, cpu=False)
                _data_loader = self.get_test_data(data_loader, _edge)
                for num, all_data in enumerate(_data_loader):
                    data = [all_data[x] for x in self.used_keys]
                    if self._cuda:
                        data = [x.to(self.device) for x in data]
                    r, p1, p2, n1, n2 = data
                    pos, neg = (p1, p2), (n1, n2)
                    for i in (0, 1):
                        feat1 = F.embedding(
                            torch.cat([pos[i].unsqueeze(-1), neg[i]], dim=-1), embs)
                        feat2 = F.embedding(pos[1-i], embs)
                        feat_args = (feat1, feat2) if i == 0 else (
                            feat2, feat1)
                        ratings = self.scoring.predict(*feat_args, relation=r)
                        lp_tracker.update(ratings, r, edges=torch.stack(
                            [p1, p2, r], dim=-1).cpu().numpy())
                    if 0 < n < len(lp_tracker):
                        break
        for metric in lp_tracker.macro:
            print({metric: lp_tracker.macro[metric]})
        lp_tracker.summarize()
        return lp_tracker

    def eval_f1auc(self, true_edges, false_edges):
        self.model.eval()
        results = []
        pred_dict = {}
        with torch.no_grad():
            for edge_type in range(self.n_edge_types):
                if edge_type not in true_edges:
                    continue
                if len(true_edges[edge_type]) == 0:
                    continue
                embs = self.get_all_rela_embedding(edge_type, cpu=False)
                pred_list = []
                for edges in (true_edges[edge_type], false_edges[edge_type]):
                    for i in range(0, len(edges), self.batch_size):
                        _edges = edges[i: i+self.batch_size]
                        _edges = torch.LongTensor(_edges)
                        r = torch.ones(len(_edges)).long() * edge_type
                        if self._cuda:
                            _edges = _edges.to(self.device)
                            r = r.to(self.device)
                        h_feat = F.embedding(_edges[:, 0], embs)
                        t_feat = F.embedding(_edges[:, 1], embs)
                        ratings = self.scoring.predict(h_feat, t_feat, r).view(-1)
                        pred_list.append(ratings)
                pred_list = torch.cat(pred_list)
                pred_dict[edge_type] = pred_list
        for edge_type, pred_list in pred_dict.items():
            num_true = len(true_edges[edge_type])
            threshold = pred_list.topk(num_true)[0][-1]
            y_pred = (pred_list >= threshold).cpu().numpy()
            y_scores = pred_list.cpu().numpy()
            y_true = np.zeros_like(y_scores)
            y_true[:num_true] = 1
            ps, rs, _ = precision_recall_curve(y_true, y_scores)
            results.append({'auc': roc_auc_score(y_true, y_scores),
                            'f1': f1_score(y_true, y_pred), 'pr': auc(rs, ps)})
        return {key: np.mean([x[key] for x in results]) for key in results[0].keys()}

    def get_test_data(self, data_loader, rela_idx):
        batch_data = []
        batch_len = 0
        _batch_len = 0
        for num, all_data in enumerate(data_loader):
            idx = all_data['r'] == rela_idx
            batch_len += idx.sum().item()
            _batch_len = len(all_data['r'])
            batch_data.append({k: v[idx] for k, v in all_data.items()})
            if batch_len > _batch_len:
                keys = batch_data[0].keys()
                yield {k: torch.cat([_batch[k] for _batch in batch_data], 0) for k in keys}
                batch_data = []
                batch_len = 0
        if len(batch_data) > 0:
            keys = batch_data[0].keys()
            yield {k: torch.cat([_batch[k] for _batch in batch_data], 0) for k in keys}

    def get_all_rela_embedding(self, target_idx, cpu=True):
        node_size = self.graph.node_size
        all_emb = []
        for i in range(0, node_size, self.batch_size):
            end_idx = min(node_size, i + self.batch_size)
            _node_idx = torch.arange(i, end_idx).long()
            if self._cuda:
                _node_idx = _node_idx.cuda()
            _target_idx = torch.ones_like(_node_idx) * target_idx
            emb = self.get_embedding(_node_idx, _target_idx)
            if cpu:
                emb = emb.cpu()
            all_emb.append(emb)
        return torch.cat(all_emb, dim=0)

    def get_embedding(self, node_idx, target_idx, **kwargs):
        old_size = node_idx.size()
        if target_idx is not None:
            target_idx = target_idx.unsqueeze(1) if len(
                target_idx.size()) != len(node_idx.size()) else target_idx
            target_idx = target_idx.expand_as(node_idx).contiguous().view(-1)
        node_idx = node_idx.contiguous().view(-1)
        return self.model(node_idx, target_idx, **kwargs).view(*old_size, -1)
