import pydot


def visualize_graph(objs, triples, hide_obj_mask, vocab):
    '''
    Visualize the graph as a graphviz rendered image

    :param objs: [num_objs]
    :param triples: [num_triples, 3] where each triple is (i, p, j),
                    such that (objs[i], p, objs[j]) is an edge
    :param hide_obj_mask: [num_objs] mask of objects to hide
    :param vocab: {"object_idx_to_name": [...], "pred_idx_to_name": [...]}
    '''
    pydot_graph = pydot.Dot(graph_type='digraph')  # , dpi=300

    for i in range(objs.size(0)):
        obj = objs[i].cpu().item()
        obj_name = vocab['object_idx_to_name'][obj]
        pydot_graph.add_node(pydot.Node(str(obj), label=obj_name, style="filled", fillcolor="gray" if hide_obj_mask[i] else "white"))

    for i in range(triples.size(0)):
        triple = triples[i].cpu().numpy()
        s, p, o = triple
        s = objs[s].cpu().item()
        o = objs[o].cpu().item()
        p_name = vocab['pred_idx_to_name'][p]
        pydot_graph.add_edge(pydot.Edge(str(s), str(o), label=p_name))

    pydot_graph.write_svg('graph.svg')
