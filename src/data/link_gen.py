import random
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import config

def _collate_fn(batch):
    batch = default_collate(batch)
    for key, tensor in batch.items():
        if key[0] != '_':
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                batch[key] = torch.stack([x.long() for x in tensor]).transpose(0, 1)
            else:
                batch[key] = tensor.long()
    return batch


class LinkGenerator:

    _non_type = '$NonType$'

    # mode = 0: train, 1: val, 2: test, 3: full (for unsupervised learning)
    def __init__(self, edge_label, nodes_of_types=None, mode=0, n_neg=5, check_neg=True, node_type_to_id=None, use_type=True, train_exist=True, shuf_ht=True, filter_edge_type=None):
        _convert_mode = {0: 'train', 1: 'train_val', 2: 'all'}
        edges_mode = _convert_mode[mode]
        if mode == 1:
            self.edges = edge_label.val_samples
            self.edges_type = edge_label.val_labels
        elif mode == 2:
            self.edges = edge_label.test_samples
            self.edges_type = edge_label.test_labels
        else:
            if len(edge_label.train_samples) == 0:
                print('using whole network')
                self.edges = edge_label.samples
                self.edges_type = edge_label.labels
                edges_mode = 'all'
            else:
                self.edges = edge_label.train_samples
                self.edges_type = edge_label.train_labels

        if filter_edge_type is not None:
            new_e, new_et = [], []
            for e, et in zip(self.edges, self.edges_type):
                if et in filter_edge_type:
                    new_e.append(e), new_et.append(et)
            self.edges, self.edges_type = new_e, new_et

        self.edges_of_types = edge_label.edges_of_types(edges_mode)
        self.use_type = use_type

        if nodes_of_types is not None:
            if train_exist:
                nodes_of_types = {node_type: list(set(nodes).intersection(edge_label.train_nodes)) for node_type, nodes in nodes_of_types.items()}
            nodes_of_types = {k: np.array(v) for k, v in nodes_of_types.items()}
            self.type_of_nodes = {}
            for node_type, nodes in nodes_of_types.items():
                for node in nodes:
                    self.type_of_nodes[node] = node_type
            if train_exist:
                n_edges = len(self.edges)
                self.edges = list(filter(lambda x: x[0] in self.type_of_nodes and x[1] in self.type_of_nodes, self.edges))
                print('#Removed edges:', n_edges - len(self.edges))
            self.node_type_to_id = node_type_to_id
            if self.node_type_to_id is None:
                self.node_type_to_id = {node_type: i for i, node_type in enumerate(sorted(nodes_of_types))}
        else:
            self.type_of_nodes = None
            self.node_type_to_id = None

        if nodes_of_types is None or not use_type:
            self.nodes_of_types = {LinkGenerator._non_type: np.array(list(edge_label.train_nodes))}
        else:
            self.nodes_of_types = nodes_of_types

        for nodes in self.nodes_of_types.values():
            random.shuffle(nodes)
        self.iter_nodes = {node_type: 0 for node_type in self.nodes_of_types}
        self.n_neg = n_neg
        self.check_neg = check_neg
        self.shuf_ht = shuf_ht

        all_edges = []
        for edges in self.edges_of_types.values():
            all_edges += edges
        self.true_head, self.true_tail = self.get_true_head_and_tail()

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        p_edge = list(self.edges[index])
        if self.shuf_ht:
            if random.random() < 0.5:
                p_edge[0], p_edge[1] = p_edge[1], p_edge[0]
        p_edge_type = self.edges_type[index]
        n_nodes = [[], []]
        p_type_edges = self.edges_of_types[p_edge_type]
        for i in (0, 1):
            node_type = self.type_of_nodes[p_edge[i]] if self.use_type else self._non_type
            if self.check_neg:

                negative_sample_list = []
                negative_sample_size = 0
                relation = p_edge_type
                head, tail = p_edge[0], p_edge[1]
                while negative_sample_size < self.n_neg:
                    negative_sample = np.random.randint(len(self.nodes_of_types[node_type]), size=self.n_neg*2)
                    negative_sample = self.nodes_of_types[node_type][negative_sample]
                    if i == 0:
                        mask = np.in1d(
                            negative_sample, 
                            self.true_head[(relation, tail)], 
                            assume_unique=True, 
                            invert=True
                        )
                    else:
                        mask = np.in1d(
                            negative_sample, 
                            self.true_tail[(head, relation)], 
                            assume_unique=True, 
                            invert=True
                        )
                    negative_sample = negative_sample[mask]
                    negative_sample_list.append(negative_sample)
                    negative_sample_size += negative_sample.size
                
                n_nodes[i] = np.concatenate(negative_sample_list)[:self.n_neg]
            else:
                n_nodes[i] = random.sample(self.nodes_of_types[node_type], self.n_neg)
        ret = {'r': p_edge_type,
               'h': p_edge[0],
               't': p_edge[1],
               }
        if self.n_neg > 0:
            ret['h_neg'] = n_nodes[0]
            ret['t_neg'] = n_nodes[1]
        if self.type_of_nodes is not None:
            ret['h_type'] = self.node_type_to_id[self.type_of_nodes[p_edge[0]]]
            ret['t_type'] = self.node_type_to_id[self.type_of_nodes[p_edge[1]]]
        return ret

    def _get_random_node(self, node_type=None):
        if node_type is None:
            node_type = LinkGenerator._non_type
        _iter = self.iter_nodes[node_type]
        if _iter == len(self.nodes_of_types[node_type]):
            random.shuffle(self.nodes_of_types[node_type])
            _iter = 0
            self.iter_nodes[node_type] = 0
        node = self.nodes_of_types[node_type][_iter]
        self.iter_nodes[node_type] += 1
        return node
    
    collate_fn = _collate_fn

    def get_true_head_and_tail(self):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for relation, edges in self.edges_of_types.items():
            for head, tail, _ in edges:
                if (head, relation) not in true_tail:
                    true_tail[(head, relation)] = []
                true_tail[(head, relation)].append(tail)
                if (relation, tail) not in true_head:
                    true_head[(relation, tail)] = []
                true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

def init_seed_fn(seed=config.seed, use_worker_id=False):
    def init_fn(worker_id):
        worker_seed = seed + worker_id if use_worker_id else seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return init_fn

def init_fn(worker_id):
    worker_seed = config.seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)