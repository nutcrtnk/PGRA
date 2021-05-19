import config

def add_reverse_edges(graph):
    tnode = graph.type_of_nodes
    to_oid = graph.look_back_list
    et2id = graph._edge_type_to_id[graph._default_type]
    id_size = graph.edge_type_size
    for i, _rev in enumerate(list(graph.rev_list)):
        if _rev == -1:
            graph.rev_list[i] = len(graph.rev_list)
            graph.rev_list.append(i)
    for k, v in list(et2id.items()):
        et2id['{}{}'.format(k, config.reverse_suffix)] = graph.rev_list[v]
    graph._edge_type_to_id[graph._default_type] = et2id
    graph.edge_type_size = len(graph.rev_list)
    for n1, n2, attr in list(graph.G.edges(data=True)):
        if attr['label'] in graph.directed_edge_types:
            graph.add_edge(to_oid[n2], to_oid[n1], directed=True, \
                label=graph.rev_list[attr['label']], \
                weight=attr['weight'], types=(tnode[n2], tnode[n1]), \
                edge_type='{}{}'.format(attr['edge_type'], config.reverse_suffix))
    return graph