#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo, Azade Farshad
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
Script to train SIMSG
"""

import argparse

import os
import math
import tqdm

import numpy as np
import torch
import torch.optim as optim

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from simsg.data import imagenet_deprocess_batch

from simsg.discriminators import PatchDiscriminator, AcCropDiscriminator, MultiscaleDiscriminator, divide_pred
from simsg.losses import get_gan_losses, gan_percept_loss, GANLoss, VGGLoss
from simsg.metrics import jaccard
from simsg.model import SIMSGModel
from simsg.model import GATModel
from simsg.utils import int_tuple
from simsg.utils import timeit, bool_flag, LossManager

from simsg.loader_utils import build_train_loaders
from scripts.train_utils import *

torch.backends.cudnn.benchmark = True

# for clevr, change to './datasets/clevr/target'
DATA_DIR = os.path.expanduser('./datasets/vg')


def argument_parser():
  # helps parsing the same arguments in a different script
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='vg', choices=['vg', 'clevr'])

  # Optimization hyperparameters
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--num_iterations', default=300000, type=int)
  parser.add_argument('--learning_rate', default=2e-4, type=float)

  # Switch the generator to eval mode after this many iterations
  parser.add_argument('--eval_mode_after', default=100000, type=int)

  # Dataset options
  parser.add_argument('--image_size', default='64,64', type=int_tuple)
  parser.add_argument('--num_train_samples', default=None, type=int)
  parser.add_argument('--num_val_samples', default=1024, type=int)
  parser.add_argument('--shuffle_val', default=True, type=bool_flag)
  parser.add_argument('--loader_num_workers', default=4, type=int)
  parser.add_argument('--include_relationships', default=True, type=bool_flag)

  parser.add_argument('--vg_image_dir', default=os.path.join(DATA_DIR, 'images'))
  parser.add_argument('--train_h5', default=os.path.join(DATA_DIR, 'train.h5'))
  parser.add_argument('--val_h5', default=os.path.join(DATA_DIR, 'val.h5'))
  parser.add_argument('--test_h5', default=os.path.join(DATA_DIR, 'test.h5'))
  parser.add_argument('--vocab_json', default=os.path.join(DATA_DIR, 'vocab.json'))
  parser.add_argument('--max_objects_per_image', default=10, type=int)
  parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

  # Generator options
  parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
  parser.add_argument('--embedding_dim', default=50, type=int)
  parser.add_argument('--gconv_dim', default=256, type=int) # 128
  parser.add_argument('--gconv_hidden_dim', default=512, type=int)
  parser.add_argument('--gconv_num_layers', default=5, type=int)
  parser.add_argument('--mlp_normalization', default='none', type=str)
  parser.add_argument('--decoder_network_dims', default='1024,512,256,128,64', type=int_tuple)
  parser.add_argument('--normalization', default='batch')
  parser.add_argument('--activation', default='leakyrelu-0.2')
  parser.add_argument('--layout_noise_dim', default=32, type=int)

  parser.add_argument('--image_feats', default=True, type=bool_flag)
  parser.add_argument('--selective_discr_obj', default=True, type=bool_flag)
  parser.add_argument('--feats_in_gcn', default=True, type=bool_flag)
  parser.add_argument('--feats_out_gcn', default=True, type=bool_flag)
  parser.add_argument('--is_baseline', default=False, type=int)
  parser.add_argument('--is_supervised', default=False, type=int)

  # Generator losses
  parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
  parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)

  # Generic discriminator options
  parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
  parser.add_argument('--gan_loss_type', default='gan')
  parser.add_argument('--d_normalization', default='batch')
  parser.add_argument('--d_padding', default='valid')
  parser.add_argument('--d_activation', default='leakyrelu-0.2')

  # Object discriminator
  parser.add_argument('--d_obj_arch',
      default='C4-64-2,C4-128-2,C4-256-2')
  parser.add_argument('--crop_size', default=32, type=int)
  parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight #was 1.0
  parser.add_argument('--ac_loss_weight', default=0.1, type=float) #was 0.1

  # Image discriminator
  parser.add_argument('--d_img_arch',
      default='C4-64-2,C4-128-2,C4-256-2')
  parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

  # Output options
  parser.add_argument('--print_every', default=500, type=int)
  parser.add_argument('--timing', default=False, type=bool_flag)
  parser.add_argument('--checkpoint_every', default=5000, type=int)
  parser.add_argument('--output_dir', default=os.getcwd())
  parser.add_argument('--checkpoint_name', default='checkpoint')
  parser.add_argument('--checkpoint_start_from', default=None)
  parser.add_argument('--restore_from_checkpoint', default=True, type=bool_flag)

  # tensorboard options
  parser.add_argument('--log_dir', default="./experiments/logs_aGCN_spade", type=str)
  parser.add_argument('--max_num_imgs', default=None, type=int)

  # SPADE options
  parser.add_argument('--percept_weight', default=0., type=float)
  parser.add_argument('--weight_gan_feat', default=0., type=float)
  parser.add_argument('--multi_discriminator', default=False, type=bool_flag)
  parser.add_argument('--spade_gen_blocks', default=False, type=bool_flag)
  parser.add_argument('--layout_pooling', default="sum", type=str)

  return parser


def build_model(args, vocab):
  if args.checkpoint_start_from is not None:
    checkpoint = torch.load(args.checkpoint_start_from)
    kwargs = checkpoint['model_kwargs']
    model = GATModel(**kwargs)
    raw_state_dict = checkpoint['model_state']
    state_dict = {}
    for k, v in raw_state_dict.items():
      if k.startswith('module.'):
        k = k[7:]
      state_dict[k] = v
    model.load_state_dict(state_dict)

  else:
    kwargs = {
      'vocab': vocab,
      'image_size': args.image_size,
      'embedding_dim': args.embedding_dim,
      'gconv_dim': args.gconv_dim,
      'gconv_hidden_dim': args.gconv_hidden_dim,
      'gconv_num_layers': args.gconv_num_layers,
      'mlp_normalization': args.mlp_normalization,
      'decoder_dims': args.decoder_network_dims,
      'normalization': args.normalization,
      'activation': args.activation,
      'mask_size': args.mask_size,
      'layout_noise_dim': args.layout_noise_dim,
      'img_feats_branch': args.image_feats,
      'feats_in_gcn': args.feats_in_gcn,
      'feats_out_gcn': args.feats_out_gcn,
      'is_baseline': args.is_baseline,
      'is_supervised': args.is_supervised,
      'spade_blocks': args.spade_gen_blocks,
      'layout_pooling': args.layout_pooling
    }

    model = GATModel(**kwargs)

  return model, kwargs


def build_obj_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_obj_weight = args.d_obj_weight
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'vocab': vocab,
    'arch': args.d_obj_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
    'object_size': args.crop_size,
  }
  discriminator = AcCropDiscriminator(**d_kwargs)

  return discriminator, d_kwargs


def build_img_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_img_weight = args.d_img_weight
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': args.d_img_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
  }

  if args.multi_discriminator:
    discriminator = MultiscaleDiscriminator(input_nc=3, num_D=2)
  else:
    discriminator = PatchDiscriminator(**d_kwargs)

  return discriminator, d_kwargs


def check_model(args, t, loader, model):

  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      imgs_src = None

      if args.dataset == "vg" or (args.dataset == "clevr" and not args.is_supervised):
        imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = batch
      elif args.dataset == "clevr":
        imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
        triple_to_img, imgs_in = batch

      model_masks = masks

      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks,
                        src_image=imgs_in, imgs_src=imgs_src)
      imgs_pred, boxes_pred, masks_pred, _, _ = model_out

      skip_pixel_loss = False
      total_loss, losses = calculate_model_losses(
                                args, skip_pixel_loss, imgs, imgs_pred,
                                boxes, boxes_pred)

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

    samples = {}
    samples['gt_img'] = imgs

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in, imgs_src=imgs_src)
    samples['gt_box_gt_mask'] = model_out[0]

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, src_image=imgs_in, imgs_src=imgs_src)
    samples['generated_img_gt_box'] = model_out[0]

    samples['masked_img'] = model_out[3][:,:3,:,:]

    for k, v in samples.items():
      samples[k] = imagenet_deprocess_batch(v)

    mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    avg_iou = total_iou / total_boxes

    masks_to_store = masks
    if masks_to_store is not None:
      masks_to_store = masks_to_store.data.cpu().clone()

    masks_pred_to_store = masks_pred
    if masks_pred_to_store is not None:
      masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  batch_data = {
    'objs': objs.detach().cpu().clone(),
    'boxes_gt': boxes.detach().cpu().clone(),
    'masks_gt': masks_to_store,
    'triples': triples.detach().cpu().clone(),
    'obj_to_img': obj_to_img.detach().cpu().clone(),
    'triple_to_img': triple_to_img.detach().cpu().clone(),
    'boxes_pred': boxes_pred.detach().cpu().clone(),
    'masks_pred': masks_pred_to_store
  }
  out = [mean_losses, samples, batch_data, avg_iou]

  return tuple(out)


def main(args):

  print(args)
  check_args(args)
  float_dtype = torch.cuda.FloatTensor

  writer = SummaryWriter(args.log_dir) if args.log_dir is not None else None

  vocab, train_loader, val_loader = build_train_loaders(args)
  model, model_kwargs = build_model(args, vocab)
  model.type(float_dtype)
  print(model)

  # use to freeze parts of the network (VGG feature extraction)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.learning_rate)

  loss = torch.nn.CrossEntropyLoss()

  restore_path = None
  if args.checkpoint_start_from is not None:
    restore_path = args.checkpoint_start_from
  else:
    if args.restore_from_checkpoint:
      restore_path = '%s_model.pt' % args.checkpoint_name
      restore_path = os.path.join(args.output_dir, restore_path)
  if restore_path is not None and os.path.isfile(restore_path):
    print('Restoring from checkpoint:')
    print(restore_path)
    checkpoint = torch.load(restore_path)

    model.load_state_dict(checkpoint['model_state'], strict=False)
    print(optimizer)
    #optimizer.load_state_dict(checkpoint['optim_state'])

    t = checkpoint['counters']['t']
    print(t, args.eval_mode_after)
    if 0 <= args.eval_mode_after <= t:
      model.eval()
    else:
      model.train()
    epoch = checkpoint['counters']['epoch']
  else:
    t, epoch = 0, 0
    checkpoint = init_checkpoint_dict(args, vocab, model_kwargs)

  while True:
    if t >= args.num_iterations:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)

    for batch in tqdm.tqdm(train_loader):
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      imgs_src = None

      if args.dataset == "vg" or (args.dataset == "clevr" and not args.is_supervised):
        imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = batch
      elif args.dataset == "clevr":
        imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
        triple_to_img, imgs_in = batch

      with timeit('forward', args.timing):
        model_boxes = boxes
        model_masks = masks

        model_out = model(objs, triples, obj_to_img,
                          boxes_gt=model_boxes, masks_gt=model_masks, src_image=imgs_in, imgs_src=imgs_src, t=t)
        nodes_pred, num_objs = model_out
        nodes_pred = nodes_pred[:num_objs]

      with timeit('loss', args.timing):
        # Skip the pixel loss if not using GT boxes
        skip_pixel_loss = (model_boxes is None)
        losses = {}
        total_loss = loss(nodes_pred, objs)

      losses['total_loss'] = total_loss.item()
      if not math.isfinite(losses['total_loss']):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      optimizer.zero_grad()
      with timeit('backward', args.timing):
        total_loss.backward()
      optimizer.step()

      if t % args.print_every == 0:
        print_G_state(args, t, losses, writer, checkpoint)

      if t % args.checkpoint_every == 0:
        print('checking on train')
        train_results = check_model(args, t, train_loader, model)
        t_losses, t_samples, t_batch_data, t_avg_iou = train_results

        checkpoint['checkpoint_ts'].append(t)
        checkpoint['train_iou'].append(t_avg_iou)

        print('checking on val')
        val_results = check_model(args, t, val_loader, model)
        val_losses, val_samples, val_batch_data, val_avg_iou = val_results

        checkpoint['val_iou'].append(val_avg_iou)

        # write images to tensorboard
        train_samples_viz = torch.cat((t_samples['gt_img'][:args.max_num_imgs, :, :, :],
                                       t_samples['masked_img'][:args.max_num_imgs, :, :, :],
                                       t_samples['generated_img_gt_box'][:args.max_num_imgs, :, :, :]), dim=3)

        val_samples_viz = torch.cat((val_samples['gt_img'][:args.max_num_imgs, :, :, :],
                                     val_samples['masked_img'][:args.max_num_imgs, :, :, :],
                                     val_samples['generated_img_gt_box'][:args.max_num_imgs, :, :, :]), dim=3)

        writer.add_image('Train samples', make_grid(train_samples_viz, nrow=4, padding=4), global_step=t)
        writer.add_image('Val samples', make_grid(val_samples_viz, nrow=4, padding=4), global_step=t)

        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)
        # write IoU to tensorboard
        writer.add_scalar('train mIoU', t_avg_iou, global_step=t)
        writer.add_scalar('val mIoU', val_avg_iou, global_step=t)
        # write losses to tensorboard
        for k, v in t_losses.items():
          writer.add_scalar('Train {}'.format(k), v, global_step=t)

        for k, v in val_losses.items():
          checkpoint['val_losses'][k].append(v)
          writer.add_scalar('Val {}'.format(k), v, global_step=t)
        checkpoint['model_state'] = model.state_dict()

        checkpoint['optim_state'] = optimizer.state_dict()
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint_path_step = os.path.join(args.output_dir,
                              '%s_%s_model.pt' % (args.checkpoint_name, str(t//10000)))
        checkpoint_path_latest = os.path.join(args.output_dir,
                              '%s_model.pt' % (args.checkpoint_name))

        print('Saving checkpoint to ', checkpoint_path_latest)
        torch.save(checkpoint, checkpoint_path_latest)
        if t % 10000 == 0 and t >= 100000:
          torch.save(checkpoint, checkpoint_path_step)


if __name__ == '__main__':
  parser = argument_parser()
  args = parser.parse_args()
  main(args)
