from itertools import chain
import pickle

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.datasets import WordNet18 as WordNet18Original


class WordNet18(WordNet18Original):
    r"""The WordNet18 dataset from the `"Translating Embeddings for Modeling
    Multi-Relational Data"
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling
    -multi-relational-data>`_ paper,
    containing 40,943 entities, 18 relations and 151,442 fact triplets,
    *e.g.*, furniture includes bed.

    This dataset extends :class:`~torch_geometric.datasets.WordNet18` dataset
    such that the list of entities and list of relations are available.

    .. note::

        The original :obj:`WordNet18` dataset suffers from test leakage, *i.e.*
        more than 80% of test triplets can be found in the training set with
        another relation type.
        Therefore, it should not be used for research evaluation anymore.
        We recommend to use its cleaned version
        :class:`~torch_geometric.datasets.WordNet18RR` instead.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/WN18')
    original_dir = 'original'
    text_dir = 'text'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[1], 'rb') as f:
            self.id2entity = pickle.load(f)
        with open(self.processed_paths[2], 'rb') as f:
            self.id2edge = pickle.load(f)
        self.entity2id = {v: k for k, v in self.id2entity.items()}
        self.edge2id = {v: k for k, v in self.id2edge.items()}

    @property
    def raw_file_names(self):
        return [
            '{}{}'.format(prefix, name)
            for prefix in ('', 'text/')
            for name in ('train.txt', 'valid.txt', 'test.txt')
        ]

    @property
    def files_to_download(self):
        return [
            ('{}/{}/{}'.format(self.url, directory, name), destination)
            for directory, destination in zip(
                (self.original_dir, self.text_dir),
                (self.raw_dir, '{}/{}'.format(self.raw_dir, self.text_dir))
            )
            for name in ('train.txt', 'valid.txt', 'test.txt')
        ]

    @property
    def processed_file_names(self):
        return ('data.pt', 'entities.pkl', 'edges.pkl')

    def download(self):
        for url, destination in self.files_to_download:
            download_url(url, destination)

    def process(self):
        srcs, dsts, edge_types = [], [], []
        id2entity, id2edge = {}, {}
        for path, text_path in zip(self.raw_paths[:3], self.raw_paths[3:]):
            with open(path, 'r') as f:
                with open(text_path, 'r') as tf:
                    data = [int(x) for x in f.read().split()[1:]]
                    data = torch.tensor(data, dtype=torch.long)
                    srcs.append(data[::3])
                    dsts.append(data[1::3])
                    edge_types.append(data[2::3])
                    text_data = tf.read().split()
                    id2entity.update({
                        k.item(): v for k, v in zip(data[::3], text_data[::3])
                    })
                    id2entity.update({
                        k.item(): v for k, v in zip(data[1::3], text_data[2::3])
                    })
                    id2edge.update({
                        k.item(): v for k, v in zip(data[2::3], text_data[1::3])
                    })

        src = torch.cat(srcs, dim=0)
        dst = torch.cat(dsts, dim=0)
        edge_type = torch.cat(edge_types, dim=0)

        train_mask = torch.zeros(src.size(0), dtype=torch.bool)
        train_mask[:srcs[0].size(0)] = True
        val_mask = torch.zeros(src.size(0), dtype=torch.bool)
        val_mask[srcs[0].size(0):srcs[0].size(0) + srcs[1].size(0)] = True
        test_mask = torch.zeros(src.size(0), dtype=torch.bool)
        test_mask[srcs[0].size(0) + srcs[1].size(0):] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        perm = (num_nodes * src + dst).argsort()

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_filter(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        with open(self.processed_paths[1], 'wb') as f:
            pickle.dump(id2entity, f)
        with open(self.processed_paths[2], 'wb') as f:
            pickle.dump(id2edge, f)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

