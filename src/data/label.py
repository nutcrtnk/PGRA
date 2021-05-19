import random
from operator import itemgetter
import config
import numpy as np
import hashlib
from data.graph import Graph
from collections import Iterable
import copy
from inspect import signature
import networkx as nx


class Label:
    modes = ['all', 'train', 'val', 'test']
    default_code = ''

    def __init__(self, data=None, ):
        self.sample_label = {k: [] for k in Label.modes}
        self.code = Label.default_code
        self.has_split = False
        if data is not None:
            for _data in data:
                sample, attr = _data[:-1], _data[-1]
                label = attr['label']
                sample = sample[0] if len(sample) == 1 else tuple(sample)
                if label is not None:
                    self.sample_label['all'].append((sample, label))
                    if 'mode' in attr and attr['mode'] is not None:
                        self.sample_label[attr['mode']].append((sample, label))
                        self.has_split = True
        self.setting = None

    def __len__(self):
        return len(self.sample_label['all'])

    def create_label_sample_dict(self, mode='all'):
        return self._create_label_sample_dict(self.sample_label[mode])

    @staticmethod
    def _create_label_sample_dict(data):
        d = {}
        for sample, labels in data:
            if isinstance(labels, int) or isinstance(labels, str):
                if labels not in d:
                    d[labels] = []
                d[labels].append(sample)
            else:
                for label in labels:
                    if label not in d:
                        d[label] = []
                    d[label].append(sample)
        return d

    def create_sample_label_dict(self, mode='all'):
        return {sample: labels for sample, labels in self.sample_label[mode]}

    @staticmethod
    def get_code(kwargs):
        if 'self' in kwargs:
            del kwargs['self']
        return hashlib.md5(str(kwargs).encode('utf-8')).hexdigest()[:8]

    def set_code(self, kwargs):
        self.code = self.get_code(kwargs)

    def save_setting(self, kwargs):
        if 'self' in kwargs:
            del kwargs['self']
        self.setting = {k: v for k, v in kwargs.items()}

    def split(self, nd_train=0, n_val=500, n_test=1000, label_type=None, seed=config.seed):
        self.set_code(locals())
        self.save_setting(locals())
        random.seed(seed)
        left_samples = set(self.samples)
        sample_label_dict = self.create_sample_label_dict()

        self.sample_label['train'] = []
        self.sample_label['val'] = []
        self.sample_label['test'] = []

        if nd_train > 0:
            label_d = self.create_label_sample_dict()
            self.sample_label['train'] = []
            for samples in label_d.values():
                for sample in random.sample(samples, nd_train):
                    self.sample_label['train'].append((sample, sample_label_dict[sample]))
        if nd_train <= 0:
            if label_type is not None:
                for sample, label in self.sample_label['all']:
                    if label == label_type:
                        continue
                    if isinstance(label, Iterable) and (label_type in label):
                        continue
                    self.sample_label['train'].append((sample, sample_label_dict[sample]))
                left_samples = left_samples - set(self.train_samples)
            if 0 < n_val < 1:
                n_val = n_val * len(left_samples)
            if 0 < n_test < 1:
                n_test = n_test * len(left_samples)
            n_test, n_val = int(n_test), int(n_val)
            train_sample = random.sample(left_samples, len(left_samples) - n_val - n_test)
            for sample in train_sample:
                self.sample_label['train'].append((sample, sample_label_dict[sample]))
            left_samples = left_samples - set(self.train_samples)
        if n_val > 0:
            val_samples = random.sample(left_samples, n_val)
            self.sample_label['val'] = [(sample, sample_label_dict[sample]) for sample in val_samples]
        else:
            self.sample_label['val'] = []
        if n_test > 0:
            left_samples = left_samples - set(val_samples)
            test_samples = random.sample(left_samples, n_test)
            self.sample_label['test'] = [(sample, sample_label_dict[sample]) for sample in test_samples]
        else:
            self.sample_label['test'] = []
        
    @staticmethod
    def _get_samples(sample_label):
        return list(map(itemgetter(0), sample_label))

    @staticmethod
    def _get_labels(sample_label):
        return list(map(itemgetter(1), sample_label))

    @property
    def samples(self):
        return self._get_samples(self.sample_label['all'])

    @property
    def labels(self):
        return self._get_labels(self.sample_label['all'])

    @property
    def train_samples(self):
        return self._get_samples(self.sample_label['train'])

    @property
    def train_labels(self):
        return self._get_labels(self.sample_label['train'])

    @property
    def val_samples(self):
        return self._get_samples(self.sample_label['val'])

    @property
    def val_labels(self):
        return self._get_labels(self.sample_label['val'])

    @property
    def test_samples(self):
        return self._get_samples(self.sample_label['test'])

    @property
    def test_labels(self):
        return self._get_labels(self.sample_label['test'])


class NodeLabel(Label):

    def __init__(self, graph, node_type=None):
        self.node_size = graph.node_size
        self.label_size = graph.label_size
        nodes = [x for x in graph.G.nodes(data=True) if
                 (node_type is None or x[1]['node_type'] == node_type) and 'label' in x[1]]
        super().__init__(nodes)

    def binarize_label(self):
        bin_labels = np.zeros((self.node_size, self.label_size))
        for sample, labels in self.sample_label['all']:
            if isinstance(labels, int):
                bin_labels[sample][labels] = 1
            else:
                for label in labels:
                    bin_labels[sample][label] = 1
        return bin_labels


class EdgeLabel(Label):

    def __init__(self, graph, edge_type=None, directed_edge_types=None):
        super().__init__(None)
        nx_graph = graph
        if isinstance(graph, Graph):
            nx_graph = graph.G
            self.directed_edge_types = graph.directed_edge_types
        else:
            self.directed_edge_types = directed_edge_types if directed_edge_types is not None else set()
        data = nx_graph.edges(data=True)
        node_type_dict = nx.get_node_attributes(nx_graph, 'node_type')
        added_undirected_edges = set()
        self.unused_edges = []
        self.mode = 'all'

        def _sorted(type1, node_idx1, type2, node_idx2):
            if type1 < type2:
                return node_idx1, node_idx2
            if type2 < type1:
                return node_idx2, node_idx1
            return sorted([node_idx1, node_idx2])
        
        if data is not None:
            for _data in data:
                edge_attr = _data[-1]
                sample, label = _data[:-1], edge_attr['label']
                sample = tuple([*sample, label])
                if label is not None:
                    if label not in self.directed_edge_types:
                        n1, n2 = sample[:2]
                        sample = tuple([*_sorted(node_type_dict[n1], n1, node_type_dict[n2], n2), *sample[2:]])
                        encoded_edge = '{},{},{}'.format(sample[0], sample[1], label)
                        if encoded_edge in added_undirected_edges:
                            continue
                        else:
                            added_undirected_edges.add(encoded_edge)
                    if edge_type is None or label == edge_type:
                        self.sample_label['all'].append((sample, label))
                        if 'mode' in edge_attr and edge_attr['mode'] is not None:
                            self.sample_label[edge_attr['mode']].append((sample, label))
                            self.mode = 'train'
                            self.has_split = True
                    else:
                        self.unused_edges.append((sample, label))
        self.train_nodes = set()
        self.edge_types = set()
        self.create_train_data()

    def edges_of_types(self, mode):
        assert mode in (self.modes + ['train_val'])
        data = copy.deepcopy(self.unused_edges)
        if mode == 'train_val':
            data += self.sample_label['train'] + self.sample_label['val']
        else:
            data += self.sample_label[mode]
        data = self._duplicate_undirected_edges(data)
        return self._create_label_sample_dict(data)

    def create_train_data(self):
        train_nodes = set()
        edge_types = set()
        for edge, label in self.sample_label[self.mode]:
            train_nodes.add(edge[0])
            train_nodes.add(edge[1])
            edge_types.add(label)
        self.train_nodes = train_nodes
        self.edge_types = edge_types

    @property
    def train_edges(self):
        return self.filter_edges(self.mode)

    def filter_edges(self, mode):
        if mode == 'train_val':
            sample_label = self.sample_label['train'] + self.sample_label['val']
        else:
            sample_label = self.sample_label[mode]
        return self._get_samples(self._duplicate_undirected_edges(sample_label + self.unused_edges))

    def duplicate_undirected_edges(self):
        self.sample_label = {k: self._duplicate_undirected_edges(v) for k, v in self.sample_label.items()}

    def _duplicate_undirected_edges(self, sample_label):
        _sample_label = []
        for edge, label in sample_label:
            _sample_label.append((edge, label))
            if label not in self.directed_edge_types:
                _sample_label.append(((edge[1], edge[0], label), label))
        return _sample_label

    def split(self, **kwargs):
        super().split(**kwargs)
        self.mode = 'train'
        self.remove_non_exist()

    @staticmethod
    def get_code(kwargs):
        return '{}_{}'.format(kwargs['n_test'], kwargs['seed'])

    def remove_non_exist(self):
        self.create_train_data()

        def check_edges(sample_label):
            checked_edges = []
            for sample, label in sample_label:
                if sample[0] in self.train_nodes and sample[1] in self.train_nodes and label in self.edge_types:
                    checked_edges.append((sample, label))
            return checked_edges

        self.sample_label['val'] = check_edges(self.sample_label['val'])
        self.sample_label['test'] = check_edges(self.sample_label['test'])
