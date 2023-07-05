#!/usr/bin/python
#
# Copyright 2020 Helisa Dhamo
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

"""
This script can be used to generate images with changes for Visual Genome for evaluation.
"""

import argparse, json
import os

import torch

from imageio import imsave

from simsg.data.visualize import visualize_graph, explore_graph, find_node, explore_graph2, explore_graph3, wn_extention_dists
from simsg.model import SIMSGModel
from simsg.model import glove
from simsg.model import GATModel
from simsg.utils import int_tuple, bool_flag
from simsg.vis import draw_scene_graph

from simsg.data.handcrafted import add_person_is_hidden

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simsg.loader_utils import build_eval_loader
from scripts.eval_utils import makedir, query_image_by_semantic_id, save_graph_json, \
  remove_duplicates, save_image_from_tensor, save_image_with_label, is_background, remove_node

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='./experiments/vg/')
parser.add_argument('--experiment', default="spade_vg", type=str)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--predgraphs', default=False, type=bool_flag)
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--save_graph_image', default=False, type=bool_flag)
parser.add_argument('--save_graph_json', default=False, type=bool_flag) # before and after
parser.add_argument('--with_query_image', default=False, type=bool)
parser.add_argument('--mode', default='remove',
                    choices=['auto_withfeats', 'auto_nofeats', 'replace', 'reposition', 'remove'])
# fancy image save that also visualizes a text label describing the change
parser.add_argument('--label_saved_image', default=True, type=bool_flag)
# used for relationship changes (reposition)
# do we drop the subject, object, or both from the original location?
parser.add_argument('--drop_obj', default=False, type=bool_flag)
parser.add_argument('--drop_subj', default=True, type=bool_flag)
# used for object replacement
# if True, the position of the original object is kept (gt) while the size (H, W) comes from the predicted box
# recommended to set to True when replacing objects (e.g. bus to car),
# and to False for background change (e.g. ocean to field)
parser.add_argument('--combined_gt_pred_box', default=True, type=bool_flag)
# use with mode auto_nofeats to generate diverse objects when features phi are masked/dropped
parser.add_argument('--random_feats', default=False, type=bool_flag)

VG_DIR = os.path.expanduser('./datasets/vg')
SPLIT = "test"
parser.add_argument('--data_h5', default=os.path.join(VG_DIR, SPLIT + '.h5'))
DATASET_DIR = './datasets/sgcustom'
parser.add_argument('--dataset_path', default=DATASET_DIR)
parser.add_argument('--data_image_dir',
        default=os.path.join(DATASET_DIR, 'images'))

args = parser.parse_args()
args.dataset = "sgcustom"

def build_model(args, checkpoint):
  model = GATModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model

def run_model(args, checkpoint, output_dir, loader=None):
  if loader is None:
    loader = build_eval_loader(args, checkpoint)

  assert args.mode in ['auto_withfeats', 'auto_nofeats', 'reposition', 'replace', 'remove']

  total_correct = 0
  total_objs = 0

  total_profession_correct = 0
  total_profession = 0

  cos_sim = torch.nn.CosineSimilarity(dim=-1)

  classes = ['chef', 'doctor', 'engineer', 'farmer', 'firefighter', 'judge', 'mechanic', 'pilot', 'police', 'waiter']
  class_embeddings = [glove[x] for x in classes]
  class_embeddings = [x.cuda() for x in class_embeddings]

  results = []

  with open('idenprof_vg_sgg.jsonl', 'w') as f_out:
    i = 0
    max_i = 10 ** 6
    print("=" * 30, "Image: ", i, "=" * 30)  # before first image is loaded
    for batch in loader:
      if i >= max_i:
        break
      i += 1

      imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in, hide_obj_mask, gt_labels, image_paths = batch
      batch = [imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in, hide_obj_mask]
      # imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in, hide_obj_mask = [x.cuda() for x in batch]
      # objs, boxes, triples, hide_obj_mask = add_person_is_hidden(loader.dataset, objs, boxes, triples)
      # imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = [
      #   x.cuda() for x in (imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in)
      # ]

      #explore_graph(objs, triples, hide_obj_mask, model.vocab)
      #explore_graph2(objs, triples, hide_obj_mask, model.vocab)
      #visualize_graph(objs, triples, hide_obj_mask, model.vocab)

      if i < max_i:
        print("=" * 30, "Image: ", i, "=" * 30)  # for the next image

      # imgs_in are masked images, imgs_gt are the original images (channels=3)

      line = json.dumps({
        "image": image_paths[0],
        "relations": relations_of(objs, triples, loader.dataset.vocab),
      })
      f_out.write(line + "\n")

  print("Done")


def relations_of(objs, triples, vocab):
  result = []
  seen = set()
  for s, p, o in triples:
    s = objs[s]
    o = objs[o]
    s_label = vocab['object_idx_to_name'][s]
    o_label = vocab['object_idx_to_name'][o]
    p_label = vocab['pred_idx_to_name'][p]
    rel = (s_label, p_label, o_label)

    if rel not in seen:
      result.append(rel)
      seen.add(rel)

  return result

def main(args):
  args.checkpoint = 'checkpoint_model_0.pt'

  output_dir = args.exp_dir + "/" + args.experiment + "/" + args.mode
  if args.with_query_image:
    output_dir = output_dir + "_query"

  got_checkpoint = args.checkpoint is not None
  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, output_dir)
  else:
    print('--checkpoint not specified')


if __name__ == '__main__':
  main(args)