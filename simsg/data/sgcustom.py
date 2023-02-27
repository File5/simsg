#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import numpy as np
import h5py
import PIL

from .utils import imagenet_preprocess, Resize

from simsg.data.wordnet import WordNet18
from scripts.preprocess_vg import build_wordnet_neighbors_dict


class CustomSceneGraphDataset(Dataset):
  wordnet = WordNet18("datasets/wordnet18")

  def __init__(self, vocab, dataset_path, h5_path, image_dir, image_size=(256, 256),
               normalize_images=True, max_objects=10, max_samples=None,
               include_relationships=True, use_orphaned_objects=True,
               mode='train', clean_repeats=True, predgraphs=False,
               n_neighbors=2):
    super(CustomSceneGraphDataset, self).__init__()

    assert mode in ["train", "eval", "auto_withfeats", "auto_nofeats", "reposition", "remove", "replace"]

    self.mode = mode

    self.image_dir = image_dir
    self.image_size = image_size

    with open(os.path.join(dataset_path, 'custom_data_info.json'), 'r') as f:
      self.data_info = json.load(f)
    self.idx_to_files = self.data_info['idx_to_files']

    self.replace_img_filepath = '/dss/dsshome1/0D/ge24tav3/checkpoints/custom_images/'
    self.image_paths = []
    for path in self.idx_to_files:
        self.image_paths.append(path.replace(self.replace_img_filepath, ''))
    #self.idx_to_classes = self.data_info['ind_to_classes']
    #self.idx_to_predicates = self.data_info['ind_to_predicates']

    with open(os.path.join(dataset_path, 'custom_prediction.json'), 'r') as f:
      self.predictions = json.load(f)

    with open(os.path.join(dataset_path, 'custom_gt.json'), 'r') as f:
      gt_labels = json.load(f)
      max_idx = max(map(int, gt_labels.keys()))
      self.gt_labels = [None] * (max_idx + 1)
      for idx, label in gt_labels.items():
        self.gt_labels[int(idx)] = label

    self.vocab = vocab
    self.num_objects = len(vocab['object_idx_to_name'])

    self.use_orphaned_objects = use_orphaned_objects
    self.max_objects = max_objects
    self.max_samples = max_samples
    self.include_relationships = include_relationships

    self.evaluating = mode != 'train'
    self.predgraphs = predgraphs

    self.n_neighbors = n_neighbors

    if self.mode == 'reposition':
      self.use_orphaned_objects = False

    self.clean_repeats = clean_repeats

    transform = [Resize(image_size), T.ToTensor()]
    if normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)

    self.wordnet_neighbors = build_wordnet_neighbors_dict(self.wordnet)

  def __len__(self):
    return len(self.idx_to_files)

  def add_triple(self, triples, s, p, o):
    if s is not None and o is not None:
      if self.clean_repeats and [s, p, o] in triples:
        return
      if self.predgraphs and s == o:
        return
      triples.append([s, p, o])

  def get_graph_data(self, index):
    box_topk = 30
    rel_topk = 20

    custom_data_info = self.data_info
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']
    custom_prediction = self.predictions

    image_idx = index

    image_path = custom_data_info['idx_to_files'][image_idx]
    boxes = custom_prediction[str(image_idx)]['bbox'][:box_topk]
    box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
    box_scores = custom_prediction[str(image_idx)]['bbox_scores'][:box_topk]
    all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
    all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
    all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

    for i in range(len(box_labels)):
        box_labels[i] = ind_to_classes[box_labels[i]]

    rel_labels = []
    rel_scores = []
    triples = []
    obj_names = box_labels
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            rel_scores.append(all_rel_scores[i])
            label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[all_rel_labels[i]] + ' => ' + str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]
            rel_labels.append(label)
            triples.append([box_labels[all_rel_pairs[i][0]], ind_to_predicates[all_rel_labels[i]], box_labels[all_rel_pairs[i][1]]])

    rel_labels = rel_labels[:rel_topk]
    rel_scores = rel_scores[:rel_topk]
    return triples, obj_names, boxes

  def get_object_idx(self, obj_name):
    DEFAULT_SUFFIX = '.n.01'
    k = obj_name
    if k not in self.vocab['object_name_to_idx']:
      k = k + DEFAULT_SUFFIX
    if k not in self.vocab['object_name_to_idx']:
      k = '__image__'
    return self.vocab['object_name_to_idx'][k]

  def get_size(self, image_size):
      min_size = 600
      max_size = 1000
      w, h = image_size
      size = min_size
      if max_size is not None:
          min_original_size = float(min((w, h)))
          max_original_size = float(max((w, h)))
          if max_original_size / min_original_size * size > max_size:
              size = int(round(max_size * min_original_size / max_original_size))
      if (w <= h and w == size) or (h <= w and h == size):
          return (w, h)
      if w < h:
          ow = size
          oh = int(size * h / w)
      else:
          oh = size
          ow = int(size * w / h)
      return (ow, oh)

  def __getitem__(self, index):
    """
    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (num_objs,)
    - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
    - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    H, W = self.image_size
    #return self.get_graph_data(index) # TODO: remove, debug only
    #return os.path.join(self.image_dir, self.image_paths[index])

    img_path = os.path.join(self.image_dir, self.image_paths[index])
    with open(img_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        #print(WW, HH)
        image = self.transform(image.convert('RGB'))

    map_overlapping_obj = {}
    graph_data, obj_names, orig_boxes = self.get_graph_data(index)
    obj_idxs = [self.get_object_idx(n) for n in obj_names]

    objs = []
    boxes = []

    obj_idx_mapping = {}
    counter = 0
    WW, HH = self.get_size((WW, HH))
    for i, obj_idx in enumerate(obj_idxs):
      curr_obj = obj_idx
      x1, y1, x2, y2 = orig_boxes[i]
      curr_box = torch.FloatTensor([x1/WW, y1/HH, x2/WW, y2/HH])

      objs.append(curr_obj)
      boxes.append(curr_box)
      map_overlapping_obj[i] = counter
      counter += 1

      obj_idx_mapping[obj_idx] = map_overlapping_obj[i]

    # Extend with WordNet objects
    seen_objs = set(objs)
    extend_triples = []
    source_objs, source_boxes = objs, boxes
    for n in range(self.n_neighbors):
      extend_objs = []
      extend_boxes = []

      for obj_idx, box in zip(source_objs, source_boxes):
        obj_name = self.vocab['object_idx_to_name'][obj_idx]
        obj_synset = self.vocab['names_to_synsets'].get(obj_name, obj_name)
        if obj_synset in self.wordnet_neighbors:
          for neighbor_synset, edge_idx in self.wordnet_neighbors[obj_synset]:
            try:
              neighbor_idx = self.vocab['object_name_to_idx'][neighbor_synset]
            except KeyError:
              try:
                neighbor_name = self.vocab['synsets_to_names'][neighbor_synset]
                neighbor_idx = self.vocab['object_name_to_idx'][neighbor_name]
              except Exception:
                continue  # further than n_neighbor
            if neighbor_idx not in seen_objs:
              extend_objs.append(neighbor_idx)
              i = len(map_overlapping_obj)
              map_overlapping_obj[i] = i
              obj_idx_mapping[neighbor_idx] = i
              extend_boxes.append(box)
              seen_objs.add(neighbor_idx)
            s = obj_idx_mapping.get(obj_idx, None)
            p = edge_idx
            o = obj_idx_mapping.get(neighbor_idx, None)
            self.add_triple(extend_triples, s, p, o)
      objs.extend(extend_objs)
      boxes.extend(extend_boxes)
      source_objs, source_boxes = extend_objs, extend_boxes

    # The last object will be the special __image__ object
    objs.append(self.vocab['object_name_to_idx']['__image__'])
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))

    num_objs = counter + 1

    triples = []
    for s_name, p_name, o_name in graph_data:
      s = self.get_object_idx(s_name)
      p = self.vocab['pred_name_to_idx'][p_name]
      o = self.get_object_idx(o_name)
      s = obj_idx_mapping.get(s, None)
      o = obj_idx_mapping.get(o, None)
      self.add_triple(triples, s, p, o)

    # Extend with WordNet triples
    triples.extend(extend_triples)

    # Add dummy __in_image__ relationships for all objects
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(num_objs - 1):
      triples.append([i, in_image, num_objs - 1])

    _, _, _, hide_obj_mask = add_person_is_hidden(self, objs, boxes, triples)
    objs = torch.LongTensor(objs)
    boxes = torch.stack(boxes)
    triples = torch.LongTensor(triples)

    gt_label = self.gt_labels[index]
    return image, objs, boxes, triples, hide_obj_mask, gt_label


def find_in_tensor(tensor, value):
  for i, v in enumerate(tensor):
    if v == value:
      return i


def add_person_is_hidden(dataset, objs, boxes, triples):
  person_idx = find_in_tensor(objs, dataset.vocab['object_name_to_idx']['person'])
  o = person_idx
  p = dataset.vocab['pred_name_to_idx']['_instance_hyponym']
  #objs = torch.cat([objs, torch.tensor([0])])
  objs.append(0)
  boxes.append(boxes[person_idx])
  s = len(objs) - 1  #dataset.vocab['object_name_to_idx']['profession']
  # o_idx = find_in_tensor(objs, o)
  # if o_idx is None:
  #   objs.append(o)
  #   boxes.append(boxes[person_idx])
  #   o_idx = len(objs) - 1
  dataset.add_triple(triples, s, p, o)
  hide_obj_mask = torch.zeros(len(objs), dtype=torch.uint8)
  hide_obj_mask[s] = 1
  hide_obj_mask = hide_obj_mask > 0
  return objs, boxes, triples, hide_obj_mask


def collate_fn_nopairs_noimgs(batch):
  """
  Collate function to be used when wrapping a HandcraftedSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, 3, H, W)
  - objs: LongTensor of shape (num_objs,) giving categories for all objects
  - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
  - triples: FloatTensor of shape (num_triples, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (num_triples,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n]
  - imgs_masked: FloatTensor of shape (N, 4, H, W)
  """
  # batch is a list, and each element is (image, objs, boxes, triples)
  all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []

  all_imgs_masked = []

  obj_offset = 0

  all_hide_obj_masks = []

  all_gt_labels = []

  for i, (img, objs, boxes, triples, hide_obj_mask, gt_label) in enumerate(batch):

    #all_imgs.append(img[None])
    num_objs, num_triples = objs.size(0), triples.size(0)

    all_objs.append(objs)
    all_boxes.append(boxes)
    all_hide_obj_masks.append(hide_obj_mask)
    triples = triples.clone()

    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset

    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
    all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))

    # prepare input 4-channel image
    # initialize mask channel with zeros
    #masked_img = img.clone()
    #mask = torch.zeros_like(masked_img)
    #mask = mask[0:1,:,:]
    #masked_img = torch.cat([masked_img, mask], 0)
    #all_imgs_masked.append(masked_img[None])

    obj_offset += num_objs

    all_gt_labels.append(gt_label)

  all_imgs_masked = torch.tensor([])

  all_imgs = torch.tensor([])
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_hide_obj_masks = torch.cat(all_hide_obj_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  return all_imgs, all_objs, all_boxes, all_triples, \
         all_obj_to_img, all_triple_to_img, all_imgs_masked, \
         all_hide_obj_masks, all_gt_labels


from simsg.model import get_left_right_top_bottom


def overlapping_nodes(obj1, obj2, box1, box2, criteria=0.7):
  # used to clean predicted graphs - merge nodes with overlapping boxes
  # are these two objects overplapping?
  # boxes given as [left, top, right, bottom]
  res = 100 # used to project box representation in 2D for iou computation
  epsilon = 0.001
  if obj1 == obj2:
    spatial_box1 = np.zeros([res, res])
    left, right, top, bottom = get_left_right_top_bottom(box1, res, res)
    spatial_box1[top:bottom, left:right] = 1
    spatial_box2 = np.zeros([res, res])
    left, right, top, bottom = get_left_right_top_bottom(box2, res, res)
    spatial_box2[top:bottom, left:right] = 1
    iou = np.sum(spatial_box1 * spatial_box2) / \
          (np.sum((spatial_box1 + spatial_box2 > 0).astype(np.float32)) + epsilon)
    return iou >= criteria
  else:
    return False


if __name__ == '__main__':
  DATA_DIR = os.path.expanduser('./datasets/vg')
  DATASET_DIR = os.path.expanduser('./datasets/sgcustom')
  vocab_json_path = os.path.join(DATA_DIR, 'vocab.json')
  with open(vocab_json_path, 'r') as f:
    vocab = json.load(f)
  dataset = CustomSceneGraphDataset(
    vocab=vocab,
    dataset_path=DATASET_DIR,
    h5_path=None,
    image_dir=os.path.join(DATASET_DIR, 'images'),
    mode='eval',
  )
  print('Loaded dataset with %d images' % len(dataset))
  i = iter(dataset)
  print(next(i))
