from collections import defaultdict

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


def explore_graph(objs, triples, hide_obj_mask, vocab):
    classes = ['chef', 'doctor', 'engineer', 'farmer', 'firefighter', 'judge', 'mechanic', 'pilot', 'police', 'waiter']
    interested_nodes = [
        "person",
        "man",
        "woman",
        "girl",
        "boy",
        "male",
        "female",
    ] + classes


    first = True
    for i in range(objs.size(0)):
        obj = objs[i].cpu().item()
        obj_name = vocab['object_idx_to_name'][obj]
        cleaned_obj_name = obj_name.split('.', 1)[0]
        if cleaned_obj_name in interested_nodes:
            if not first:
                print(', ', end='')
            first = False
            print(obj_name, "hidden" if hide_obj_mask[i] else "", end='')
    print()


def bfs(graph, start, goal):
    visited, queue = set(), [[start]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            for adjacent in graph.get(node, []):
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)
            visited.add(node)
    return []


def find_node(objs, triples, hide_obj_mask, vocab, node_name):
    for i in range(objs.size(0)):
        obj = objs[i].cpu().item()
        obj_name = vocab['object_idx_to_name'][obj]
        cleaned_obj_name = obj_name.split('.', 1)[0]
        if cleaned_obj_name.startswith(node_name):
            return obj_name
    return None


def find_edges(objs, triples, hide_obj_mask, vocab, target_s, target_o):
    res = []
    for i in range(triples.size(0)):
        triple = triples[i].cpu().numpy()
        s, p, o = triple
        s = objs[s].cpu().item()
        o = objs[o].cpu().item()
        s_name = vocab['object_idx_to_name'][s]
        o_name = vocab['object_idx_to_name'][o]
        if s_name == target_s and o_name == target_o:
            res.append(vocab['pred_idx_to_name'][p])
    return res


def explore_graph2(objs, triples, hide_obj_mask, vocab):
    node_labels = {}
    for i in range(objs.size(0)):
        obj = objs[i].cpu().item()
        obj_name = vocab['object_idx_to_name'][obj]
        node_labels[obj] = obj_name

    p_names = {}
    edges = []
    for i in range(triples.size(0)):
        triple = triples[i].cpu().numpy()
        s, p, o = triple
        s = objs[s].cpu().item()
        o = objs[o].cpu().item()
        p_name = vocab['pred_idx_to_name'][p]
        if p_name not in p_names:
            p_names[p] = p_name
        edges.append((node_labels[s], p_names[p], node_labels[o]))

    graph = defaultdict(list)
    for s, p, o in edges:
        graph[s].append(o)
        graph[o].append(s)

    helmet = find_node(objs, triples, hide_obj_mask, vocab, 'helmet')
    print(f"helmet: {helmet}")
    print(bfs(graph, 'person', helmet))
    print(bfs(graph, 'person', 'engineer.n.01'))
    print(bfs(graph, 'person', 'woman'))
    print(find_edges(objs, triples, hide_obj_mask, vocab, 'person', 'helmet'))
