from datasets import load_dataset

dataset = load_dataset("conceptnet5", "conceptnet5")

dataset_en = {
    "train": dataset["train"].filter(lambda x: x["lang"] == "en"),
}


def triple_from_row(row):
    arg1 = row['arg1']
    rel = row['rel']
    arg2 = row['arg2']

    rel_parts = rel.split("/")
    rel_name = rel_parts[2]

    arg1_parts = arg1.split("/")
    arg1_name = arg1_parts[3]

    arg2_parts = arg2.split("/")
    arg2_name = arg2_parts[3]

    return arg1_name, rel_name, arg2_name


def iter_triples(dataset):
    for row in dataset:
        yield triple_from_row(row)


def build_conceptnet_neighbors_dict(dataset):
    neighbors = {}
    print("Building neighbors dict...")
    print("0 /", dataset.num_rows)
    i = 0
    for arg1, rel, arg2 in iter_triples(dataset):
        n = neighbors.setdefault(arg1, [])
        n.append((arg2, rel))
        if i % 100000 == 0:
            print(i, "/", dataset.num_rows)
        i += 1
    return neighbors


conceptnet_neightbors = build_conceptnet_neighbors_dict(dataset_en["train"])
