import networkx as nx
import numpy as np
import functools
from collections import OrderedDict
import copy


class Graph(object):

    _default_type = '_'

    def __init__(self):
        self.G = None
        self.node_size = 0
        self.label_size = 0
        self.edge_type_size = 0
        self.node_to_id = {}
        self.label_to_id = {}
        self.use_one_hot = True
        self.name = None
        self._edge_type_to_id = {}
        self.directed_edge_types = set()
        self.rev_list = []
        self.has_split = False

    @property
    def edge_type_to_id(self):
        return dict(self._edge_type_to_id[self._default_type])


    @property
    def look_back_list(self):
        look_back = [None for _ in range(self.node_size)]
        for node_type, nodes in self.node_to_id.items():
            for oid, nid in nodes.items():
                look_back[nid] = (node_type, oid)
        return look_back

    @property
    def look_up_dict(self):
        return {i: i for i in range(self.node_size)}

    @property
    def nodes_of_types(self):
        return {node_type: np.array(list(mapper.values())) for node_type, mapper in self.node_to_id.items()}

    @property
    def type_of_nodes(self):
        type_of_nodes = {}
        nodes_of_types = self.nodes_of_types
        for node_type, nodes in nodes_of_types.items():
            for node in nodes:
                type_of_nodes[node] = node_type
        return type_of_nodes

    def convert_to_id(self, old_id, id_type=None, add=False, mapper=None, counter=None):
        assert mapper is not None
        assert counter is not None
        mapper = getattr(self, mapper)
        if id_type is None:
            id_type = self._default_type
        if add and id_type not in mapper:
            mapper[id_type] = {}
        if add and old_id not in mapper[id_type]:
            i = getattr(self, counter)
            mapper[id_type][old_id] = i
            setattr(self, counter, i+1)
        return mapper[id_type][old_id]

    node2id = functools.partialmethod(convert_to_id, mapper='node_to_id', counter='node_size')
    label2id = functools.partialmethod(convert_to_id, mapper='label_to_id', counter='label_size')
    et2id = functools.partialmethod(convert_to_id, mapper='_edge_type_to_id', counter='edge_type_size')

    def read_adjlist(self, filename, types=None):
        if types is None:
            types = (self._default_type, self._default_type)
        if self.G is None:
            self.G = nx.DiGraph()
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            data = l.split()
            src, dst = data[0], data[1]
            for dst in data[1:]:
                self.add_edge(src, dst, 1., directed=True, types=types)
        fin.close()
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0

    @staticmethod
    def name_edge_label(types):
        return ''.join(map(str, types))

    def add_edge(self, src, dst, weight=1., directed=False, types=None, edge_type=None, node_exist=False, is_id=False, **edge_attr):
        if types is None:
            types = (self._default_type, self._default_type)
        types = list(types)
        if types[0] is None: types[0] = self._default_type
        if types[1] is None: types[1] = self._default_type
        if not is_id:
            try:
                src = self.node2id(src, id_type=types[0], add=not node_exist)
                dst = self.node2id(dst, id_type=types[1], add=not node_exist)
            except:
                return
            if src not in self.G.nodes:
                self.G.add_node(src, node_type=types[0])
            if dst not in self.G.nodes:
                self.G.add_node(dst, node_type=types[1])

        _edge_type = self.name_edge_label(types) if edge_type is None else edge_type
        _edge_id = self.et2id(_edge_type, add=True)
        self.rev_list += [-1] * (self.edge_type_size - len(self.rev_list))
        if not directed:
            self.rev_list[_edge_id] = _edge_id
        _edge_attr = copy.copy(edge_attr)
        _edge_attr.update({'label': _edge_id, 'edge_type': _edge_type, 'weight': weight})
        self.G.add_edge(src, dst, key=_edge_id, **_edge_attr)
        if not directed:
            _edge_type = self.name_edge_label(reversed(types)) if edge_type is None else edge_type
            _edge_attr = copy.copy(edge_attr)
            _edge_attr.update({'label': _edge_id, 'edge_type': _edge_type, 'weight': weight})
            self.G.add_edge(dst, src, key=_edge_id, **_edge_attr)
            if edge_type is None:
                self._edge_type_to_id[self._default_type][_edge_type] = _edge_id
        else:
            self.directed_edge_types.add(_edge_id)

    def read_edgelist(self, filename, directed=False, types=None, edge_type=None):
        if types is None:
            types = (self._default_type, self._default_type)
        if self.G is None:
            self.G = nx.MultiDiGraph()
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            data = l.split()
            src, dst = data[0], data[1]
            w = 1.
            if len(data) == 3:
                w = float(data[2])
            self.add_edge(src, dst, w, directed, types, edge_type=edge_type)
        fin.close()

    def read_node_label(self, filename, node_type=None):
        fin = open(filename, 'r')
        if node_type is None:
            node_type = self._default_type
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            labels = vec[1:]
            self.add_node_label(vec[0], labels, node_type=node_type)
        fin.close()

    def add_node_label(self, node, labels, node_type=None):
        try:
            v = self.node2id(node, id_type=node_type)
        except:
            return False
        self.G.nodes[v]['label'] = list(map(lambda x: self.label2id(x, id_type=node_type, add=True), labels))
        return True

    def read_node_features(self, filename, node_type=None):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            feature = np.array([float(x) for x in vec[1:]])
            self.add_node_feature(vec[0], feature)
        fin.close()
        self.use_one_hot = False

    def add_node_feature(self, node, feature, node_type=None):
        try:
            v = self.node2id(node, id_type=node_type)
        except:
            return False
        self.G.nodes[v]['feature'] = feature
        self.use_one_hot = False
        return True

    def add_attr(self, node, attr, value, node_type=None):
        try:
            v = self.node2id(node, id_type=node_type)
        except:
            return False
        self.G.nodes[v][attr] = value
        return True

    def get_nodes_features(self):
        if self.use_one_hot:
            return np.diag(np.ones(self.node_size))
        feat_dict = {i: self.G.nodes[i]['feature'] for i in range(self.node_size) if self.G.nodes[i]['feature'] is not None}
        feats = np.zeros([self.node_size, len(feat_dict[next(iter(feat_dict.keys()))])])
        for i, feat in feat_dict.items():
            feats[i] = feat
        return feats

    def generate_one_hot(self):
        print('generating one hot')
        self.use_one_hot = True

    def sort(self):
        n_node = 0
        remap = {}
        for node_type in sorted(self.node_to_id):
            mapper = self.node_to_id[node_type]
            new_map = {}
            for old_id in sorted(mapper):
                remap[mapper[old_id]] = n_node
                new_map[old_id] = n_node
                n_node += 1
            self.node_to_id[node_type] = new_map
        self.G = nx.relabel_nodes(self.G, remap)
        return remap

    @property
    def noftypes(self):
        return OrderedDict((node_type, len(self.node_to_id[node_type])) for node_type in sorted(self.node_to_id.keys()))
