from data import graph
import config
import networkx as nx
import re
import pickle
import numpy as np
import scipy.io as sio

map_type = {
    'author': 'a',
    'paper': 'p',
    'conference': 'c',
    'conf': 'c',
    'term': 't',
    'ref': 'r',
    'reference': 'r',

    'business': 'b',
    'location': 'l',
    'user': 'u',
    'category': 'c',

    'movie': 'm',
    'group': 'g',
    'director': 'd',
    'actor': 'a',
    'type': 't',

    'book': 'b',
    'publisher': 'p',
    'year': 'y',

    'item': 'i',
}

def _read_hin_splited(file, g=None, mode=None, pattern='r h t', directed=False, add_zero_weight=False):
    edges = __read_hin_splited(file, pattern, directed)
    if g is None:
        g = graph.Graph()
        g.G = nx.MultiDiGraph()
        
    for d in edges:
        if add_zero_weight or d['w'] != 0:
            g.add_edge(d['h'], d['t'], d['w'], d['d'], (d['a'], d['b']), edge_type=d['r'], mode=mode)
    return g

# a, b = node types of h, t
def __read_hin_splited(file, pattern='r h t', directed=False):
    symbols = ('h', 'r', 't', 'a', 'b', 'd')
    _pattern = ''
    appear = []
    for _w in pattern:
        if _w in symbols:
            _pattern += "(?P<{}>\S+)".format(_w)
            appear.append(_w)
        elif _w == 'w':
            _pattern += "(?P<{}>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)".format(_w)
            appear.append(_w)
        else:
            _pattern += _w
    pattern = re.compile(_pattern)
    edges = []
    with open(file) as f:
        for line in f:
            m = pattern.match(line.strip())
            d = {k: None for k in symbols}
            d['d'] = directed
            d['w'] = 1
            for _w in appear:
                # nonlocal h, r, t, ht, tt, d, w
                d[_w] = m.group(_w)
            d['w'] = float(d['w'])
            edges.append(d)
    return edges

def read_hin_splited(dataset, read_feature=False):
    print('Read graph')
    path = config.hin_dir / dataset
    g = _read_hin_splited(path / 'train_hin.txt', mode='train', pattern='r a b h t')
    g = _read_hin_splited(path / 'valid_mrr.txt', g, mode='val', pattern='r a b h t *?')
    g = _read_hin_splited(path / 'test_mrr.txt', g, mode='test', pattern='r a b h t *?')
    if (path / 'label.txt').exists():
        with (path / 'label.txt').open() as f:
            for line in f:
                node, node_type, labels = line.strip().split(' ')
                labels = list(labels.split(','))
                g.add_node_label(node, labels, node_type=node_type)    
    g.name = dataset
    og_path = config.hin_dir / dataset.split('_')[0]
    if (og_path / 'valid.txt').exists():
        g = _read_hin_splited(og_path / 'valid.txt', g, mode='val', pattern='r h t w')
    if (og_path / 'test.txt').exists():
        g = _read_hin_splited(og_path / 'test.txt', g, mode='test', pattern='r h t w')

    if read_feature:
        if (og_path / 'feature.txt').exists():
            print('Read feature')
            g.read_node_features(og_path / 'feature.txt')
        for f in sorted(og_path.glob('*_feature.dat')):
            node_type = (f.stem).split('_')[0]
            g.read_node_label(f, map_type[node_type])
            print('Read feature:', node_type)
    valid_data = read_hin_test(g, path / 'valid_mrr.txt')
    test_data = read_hin_test(g, path / 'test_mrr.txt')
    return g, valid_data, test_data

def read_hin_test(g, filename):
    test_data = []
    with open(filename, 'r') as f:
        d = {}
        for i, line in enumerate(f):
            data = line.strip().split(' ')
            r, ht, tt, h, t = data[:5]
            assert data[5] in ('-1', '1')
            if i % 2 == 0:
                d = {
                    'r': g.et2id(r),
                    'h': g.node2id(h, ht),
                    't': g.node2id(t, tt),
                }
            if data[5] == '1':
                d['t_neg'] = [g.node2id(n, tt) for n in data[6:]]
            else:
                d['h_neg'] = [g.node2id(n, ht) for n in data[6:]]
            if i % 2 == 1:
                test_data.append(d)
    return test_data


ds_metapaths = {
    'attr_amazon': [ 'IVI', 'IBI', 'ITI' ],
    'attr_dblp': [ 'PAP', 'PPrefP', 'PTP' ],
}


def read_dmgi(ds):
    print('reading dmgi pickle data')
    metapaths = None
    if ds.lower() == 'acm':
        data = sio.loadmat(str(config.hin_dir / 'ACM.mat'))
        metapaths = ['PAP', 'PLP']
    else:
        with open(config.hin_dir / '{}.pkl'.format(ds), 'rb') as f:
            data = pickle.load(f)
    g = graph.Graph()
    g.G = nx.MultiDiGraph()
    keys = ['label', 'train_idx', 'val_idx', 'test_idx', 'feature']
    label, train_idx, val_idx, test_idx, feature = (data[x] for x in keys)

    if ds in ds_metapaths:
        data_keys = ds_metapaths[ds]
    else:
        data_keys = list(sorted(data.keys()))
    for key in data_keys:
        if key not in keys:
            if metapaths is not None and key not in metapaths:
                continue
            if key[0] == '_':
                continue
            for n1, n2 in zip(*np.array(data[key]).nonzero()):
                g.add_edge(n1, n2, 1., edge_type=key)
    for i, feat in enumerate(feature):
        g.add_node_feature(i, feat)
    for i, _label in enumerate(label):
        g.add_node_label(i, _label.nonzero()[0])
    for node in train_idx.flatten():
        g.add_attr(node, 'mode', 'train')
    for node in val_idx.flatten():
        g.add_attr(node, 'mode', 'val')
    for node in test_idx.flatten():
        g.add_attr(node, 'mode', 'test')
    print('finish reading')
    return g

def read_hin(ds='DBLP', verbose=1, sort=True):
    path = config.hin_dir / ds
    if ds in ('youtube', 'amazon', 'twitter'):
        g = _read_hin_splited(path / 'train.txt', mode='train', pattern='r h t')
        g = _read_hin_splited(path / 'valid.txt', g, mode='val', pattern='r h t w')
        g = _read_hin_splited(path / 'test.txt', g, mode='test', pattern='r h t w')
        g.name = ds
        if (path / 'feature.txt').exists():
            g.read_node_features(path / 'feature.txt')
        g.has_split = True
    else:
        def get_types(f_name):
            return f_name.split('_')

        suffix = 'dat'
        g = graph.Graph()
        label_files = []
        for f in sorted(path.glob('*.' + suffix)):
            edge_type = None
            if verbose:
                print(f.stem)
            f_types = get_types(f.stem)
            if len(f_types) == 3:
                edge_type = f_types[2]
                f_types = f_types[:2]
            elif len(f_types) != 2:
                continue
            if f_types[0] in map_type:
                if f_types[1] == 'label':
                    label_files.append(f)
                    continue
                elif f_types[1] not in map_type:
                    continue
            else:
                continue
            if verbose:
                print(f_types)
            types = list(map(lambda x: map_type[x], f_types))
            if edge_type is None:
                edge_type = ''.join(types)
            g.read_edgelist(f, types=types, edge_type=edge_type)
        
        for f in label_files:
            if verbose:
                print('read label:', f)
            f_types = get_types(f.stem)
            g.read_node_label(f, map_type[f_types[0]])
        if sort:
            g.sort()
    g.name = ds
    return g
