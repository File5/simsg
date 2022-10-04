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
from collections import namedtuple

import torch

from imageio import imsave

from simsg.model import SIMSGModel
from simsg.model import glove
from simsg.model import GATModel
from simsg.utils import int_tuple, bool_flag
from simsg.vis import draw_scene_graph

from simsg.data.handcrafted import add_person_is_hidden

import cv2
import numpy as np

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
parser.add_argument('--data_image_dir',
        default=os.path.join(VG_DIR, 'images'))

args = parser.parse_args()
args.dataset = "vg"

HideArgs = namedtuple('HideArgs', ['hide_obj_prob'])
hide_args = HideArgs(hide_obj_prob=0.5)
def hide_nodes(args, objs):
  prob = args.hide_obj_prob
  return torch.rand(objs.size()) < prob

def build_model(args, checkpoint):
  model = GATModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model

def run_model(args, checkpoint, output_dir, loader=None):
  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_eval_loader(args, checkpoint)

  assert args.mode in ['auto_withfeats', 'auto_nofeats', 'reposition', 'replace', 'remove']

  total_correct = 0
  total_objs = 0

  cos_sim = torch.nn.CosineSimilarity(dim=-1)

  class_embeddings = model.obj_embeddings

  len_loader = len(loader)
  for i, batch in enumerate(loader):

    imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = batch
    batch = [imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in]
    imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = [x.cuda() for x in batch]
    # objs, boxes, triples, hide_obj_mask = add_person_is_hidden(loader.dataset, objs, boxes, triples)
    # imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = [
    #   x.cuda() for x in (imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in)
    # ]
    hide_obj_mask = hide_nodes(hide_args, objs)

    model_out = model(objs, triples, hide_obj_mask=hide_obj_mask)

    nodes_pred, num_objs, classification_scores = model_out
    nodes_pred = nodes_pred[:num_objs]
    classification_scores = classification_scores[:num_objs]
    node_classes_preds = torch.argmax(classification_scores, dim=1)
    correct = torch.sum(node_classes_preds[hide_obj_mask] == objs[hide_obj_mask]).item()  # only count hidden nodes
    total = torch.sum(hide_obj_mask).item()
    total_correct += correct
    total_objs += total
    # print(node_classes_preds)
    # profession_node_emb = nodes_pred[-1]
    # classes_dists = [cos_sim(profession_node_emb, c) for c in class_embeddings]
    # pred = torch.argmax(torch.tensor(classes_dists)).item()
    # gt_label = 
    # print(f"Prediction for hidden node: ({pred})")
    # print(f"Ground truth: {gt_label}")
    # print([x.item() for x in classes_dists])
    print(f"{i:4d}/{len_loader}", end="\r")
  print("")
  print("Accuracy: ", total_correct / total_objs, f" ({total_correct}/{total_objs})")


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
