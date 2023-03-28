#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo, Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except impliance with the License.
# You may obtain a copy of the License
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sg2im.decoder import DecoderNetwork
from sg2im.layout import boxes_to_layout

from simsg.graph2 import GraphTripleConv, GraphTripleConvNet, DisenTripletGCN, FactorTripletGCN, build_mlp, GAT

import torchvision as T


class GraphAEModel(nn.Module):
    """
    SIMSG network. Given a source image and a scene graph, the model generates
    a manipulated image that satisfies the scene graph constellations
    """
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none', layout_noise_dim=0,
                 img_feats_branch=False, feat_dims=128,
                 feats_in_gcn=False, feats_out_gcn=True, layout_pooling="sum",
                 gcn_mode="GCN", gat_layers=None, decoder_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2', train_sg2im=False, p_mask_eval=0.2, embedding="normal", pretrained_graph=False, **kwargs):

        super(GraphAEModel, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab
        #print(vocab)
        #assert False
        self.image_size = image_size
        self.layout_noise_dim = layout_noise_dim

        self.feats_in_gcn = feats_in_gcn
        self.feats_out_gcn = feats_out_gcn
        self.embedding_dim = embedding_dim
        self.latent_dim = int(gconv_hidden_dim / 8)
        #self.spade_blocks = spade_blocks
        self.gat_layers = gat_layers
        self.train_sg2im = train_sg2im
        self.pretrained_graph = pretrained_graph
        #self.gen_graphs = gen_graphs
        if "glove" in embedding:
            self.glove = True
        else:
            self.glove = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.image_feats_branch = img_feats_branch

        self.layout_pooling = layout_pooling

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])

        if self.glove:
            vocab_glove = self.get_glove_vocab(embedding)

            self.obj_embeddings = self.get_glove_weights(vocab, vocab_glove, num_objs +1).to(self.device)
        #self.pred_embeddings = self.get_glove_weights(vocab, vocab_glove, num_preds)
        else:
            self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim).to(self.device)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim).to(self.device)

        self.node_classifier = nn.Linear(embedding_dim, num_objs)
        self.rel_classifier = nn.Linear(embedding_dim, num_preds)

        if self.feats_in_gcn:
            gconv_input_dims = embedding_dim + 4 + feat_dims
        else:
            gconv_input_dims = embedding_dim + 4


        self.gcn_type = gcn_mode  # "FactorGCN" #or GCN, DisenGCN, FactorGCN

        class Args:
            def __init__(self):
                super(Args, self).__init__()
                self.input_dim_obj = gconv_input_dims
                self.input_dim_pred = embedding_dim
                self.hidden_dim = gconv_hidden_dim
                self.out_dim = gconv_dim

            def set_disengcn(self):
                self.nlayer = 10  # 5 #gconv_num_layers
                self.ncaps = 14  # 7
                self.nhidden = 16  # 16 #gconv_hidden_dim
                self.routit = 12  # 6
                self.dropout = 0  # 0.35

            def set_factorgnn(self):
                self.num_layers = 2
                self.num_hidden = 64
                self.num_latent = 4
                self.in_drop = 0.2
                self.residual = False

            def set_gat(self):
                #self.layers = ['gat', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn']
                self.layers = ['gat', 'gat', 'gat', 'gat', 'gat', 'gat']
                #self.layers = gat_layers
                self.use_obj_info = True
                self.use_rel_info = True
                self.k_update_steps = 1
                self.update_relations = True
                self.hidden_dim = gconv_hidden_dim
                self.output_dim = int(gconv_hidden_dim / 8) #gconv_dim
                self.gconv_pooling = gconv_pooling
                self.mlp_normalization = mlp_normalization


        hyperpm = Args()

        if self.gcn_type == "GCN":
            print("Using GCN")
            if gconv_num_layers == 0:
                self.gconv = nn.Linear(gconv_input_dims, gconv_dim)
            elif gconv_num_layers > 0:
                gconv_kwargs = {
                    'input_dim_obj': gconv_input_dims,
                    'input_dim_pred': embedding_dim,
                    'output_dim': gconv_dim,
                    'hidden_dim': gconv_hidden_dim,
                    'pooling': gconv_pooling,
                    'mlp_normalization': mlp_normalization,
                }
                self.gconv = GraphTripleConv(**gconv_kwargs)
                self.gconv = self.gconv.to(self.device)

            self.gconv_net = None
            if gconv_num_layers > 1:
                gconv_kwargs = {
                    'input_dim_obj': gconv_dim,
                    'input_dim_pred': gconv_dim,
                    'hidden_dim': gconv_hidden_dim,
                    'output_dim': self.latent_dim, #wasn't there
                    'pooling': gconv_pooling,
                    'num_layers': gconv_num_layers - 1,
                    'mlp_normalization': mlp_normalization,
                }
                self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

            graph_ae = not train_sg2im
            self.gconv_decoder = None
            if graph_ae:
                gconv_kwargs = {
                    'input_dim_obj': self.latent_dim,
                    'input_dim_pred': self.latent_dim,
                    'hidden_dim': gconv_hidden_dim,
                    'output_dim': embedding_dim,
                    'pooling': gconv_pooling,
                    'num_layers': gconv_num_layers - 1,
                    'mlp_normalization': mlp_normalization,
                }
                self.gconv_decoder = GraphTripleConvNet(**gconv_kwargs)

        elif self.gcn_type == "DisenGCN":
            print("Using DisenGCN")

            hyperpm.set_disengcn()

            self.gconv_net = DisenTripletGCN(hyperpm)

        elif self.gcn_type == "FactorGCN":
            hyperpm.set_factorgnn()
            print("Using FactorGCN")
            self.gconv_net = FactorTripletGCN(hyperpm)
        elif self.gcn_type == "GAT":
            hyperpm.set_gat()
            print("Using GAT")
            self.gconv_net = GAT(hyperpm)

            gconv_kwargs = {
                'input_dim_obj': self.latent_dim,
                'input_dim_pred': self.latent_dim,
                'hidden_dim': gconv_hidden_dim,
                'output_dim': embedding_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_decoder = GraphTripleConvNet(**gconv_kwargs)
        else:
            raise

        distributed = False
        if distributed:
            self.gconv_net = nn.DataParallel(self.gconv_net, device_ids=[0, 1])
        self.gconv_net = self.gconv_net.to(self.device)

        box_net_dim = 4
        if train_sg2im:
            box_net_layers = [self.latent_dim, gconv_hidden_dim, box_net_dim] #was gconv_dim
        else:
            box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization).to(self.device)

        if self.image_feats_branch:
            self.conv_img = nn.Sequential(
                nn.Conv2d(4, layout_noise_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(layout_noise_dim),
                nn.ReLU()
            ).to(self.device)

        self.high_level_feats = self.build_obj_feats_net().to(self.device)
        # freeze vgg
        for param in self.high_level_feats.parameters():
            param.requires_grad = False

        self.high_level_feats_fc = self.build_obj_feats_fc(feat_dims).to(self.device)

        if self.feats_in_gcn:
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim + 4 + feat_dims).to(self.device)
            if self.feats_out_gcn:
                self.layer_norm2 = nn.LayerNorm(normalized_shape=gconv_dim + feat_dims).to(self.device)
        else:
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim + 4).to(self.device)
            self.layer_norm2 = nn.LayerNorm(normalized_shape=gconv_dim + feat_dims).to(self.device)

        if self.train_sg2im:
            if self.pretrained_graph:
                print("Freezing graph encoder")
                for param in self.gconv.parameters():
                    param.requires_grad = False

                for param in self.gconv_net.parameters():
                    param.requires_grad = False

            ref_input_dim = self.latent_dim + layout_noise_dim # gconv_dim
            spade_blocks = True
            decoder_kwargs = {
                'dims': (ref_input_dim,) + decoder_dims,
                'normalization': normalization,
                'activation': activation,
                'spade_blocks': spade_blocks,
                'source_image_dims': layout_noise_dim
            }

            rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_preds]
            self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

            self.decoder_net = DecoderNetwork(**decoder_kwargs).to(self.device)

        self.p = 0.25
        self.p_box = 0.35
        self.p_obj = 0.5
        if train_sg2im:
            self.p_mask_eval = 0
            self.p_obj = 0
        else:
            self.p_mask_eval = p_mask_eval
        print("Masking percentage:", self.p_mask_eval)


    def build_obj_feats_net(self):
        # get VGG16 features for each object RoI
        vgg_net = T.models.vgg16(pretrained=True).to(self.device)
        layers = list(vgg_net.features._modules.values())[:-1]

        img_feats = nn.Sequential(*layers)

        return img_feats

    def build_obj_feats_fc(self, feat_dims):
        # fc layer following the VGG16 backbone
        return nn.Linear(512 * int(self.image_size[0] / 64) * int(self.image_size[1] / 64), feat_dims)


    def get_glove_vocab(self, glove_type):
        vocab_glove, embeddings = {}, []

        if glove_type == "glove":
            file_name = './glove/glove.6B.100d.txt'
        else:
            file_name = './glove/glove.840B.300d.txt'

        with open(file_name, 'rt') as fi:
            full_content = fi.read().strip().split('\n')

        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab_glove[i_word] = i_embeddings

        return vocab_glove

    def get_glove_weights_840B(self, vocab, vocab_glove, num_instance, type="object"): #type, object or pred

        assert type in ["object", "pred"]

        coco_embeddings = {}

        for key, _ in vocab[type + "_name_to_idx"].items():
            # print(key)
            if key == "__image__":
                coco_embeddings[key] = vocab_glove["background"]  # or unknown
                continue
            elif key == "stop sign":
                coco_embeddings[key] = vocab_glove["sign"]
                continue
            elif key == "sports ball":
                coco_embeddings[key] = vocab_glove["Sportsball"]
                continue
            elif key == "baseball glove":
                coco_embeddings[key] = vocab_glove["baseball-related"]
                continue
            elif key == "tennis racket":
                coco_embeddings[key] = vocab_glove["Tennis_rackets"]
                continue
            #Houseplant
            elif key == "potted-plant":
                coco_embeddings[key] = vocab_glove["Houseplant"]
                continue
            elif key == "NONE":
                coco_embeddings[key] = vocab_glove["unknown"]
                continue
            elif " " in key:
                target_key = key.replace(" ", "-")
                coco_embeddings[key] = vocab_glove[target_key]  # or unknown
                continue
            coco_embeddings[key] = vocab_glove[key]


        embedding_weights = []

        for i in range(len(vocab[type + "_name_to_idx"].items())):
            emb = coco_embeddings[vocab[type + "_idx_to_name"][i]]
            embedding_weights.append(emb)

        obj_embedding_weights = np.stack(embedding_weights)

        print("before size: ", num_instance, "after size: ", obj_embedding_weights.shape,)

        obj_embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(obj_embedding_weights).float())
        obj_embedding.weight.requires_grad = False

        return obj_embedding #, pred_embedding

    def get_glove_weights(self, vocab, vocab_glove, num_instance, type="object"): #type, object or pred

        assert type in ["object", "pred"]

        vg_embeddings = {}

        for key, _ in vocab[type + "_name_to_idx"].items():
            # print(key)
            if key == "__image__":
                vg_embeddings[key] = vocab_glove["background"]  # or unknown
                continue
            elif key == "stop sign":
                vg_embeddings[key] = vocab_glove["sign"]
                continue
            elif key == "sports ball":
                vg_embeddings[key] = vocab_glove["Sportsball"]
                continue
            elif key == "baseball glove":
                vg_embeddings[key] = vocab_glove["baseball-related"]
                continue
            elif key == "tennis racket":
                vg_embeddings[key] = vocab_glove["Tennis_rackets"]
                continue
            elif key == "potted plant":
                vg_embeddings[key] = vocab_glove["Houseplant"]
                continue
            elif key == "NONE":
                vg_embeddings[key] = vocab_glove["unknown"]
                continue
            elif " " in key:
                target_key = key.replace(" ", "-")
                vg_embeddings[key] = vocab_glove[target_key]  # or unknown
                continue
            vg_embeddings[key] = vocab_glove[key]

        vg_embeddings["NONE"] = vocab_glove["unknown"]
        # for key, _ in vocab["pred_name_to_idx"].items():
            # print(key)
            # if key == "__in_image__":
            #     vg_embeddings[key] = vocab_glove["in"]  # or unknown
            #     continue
            # elif key == "next to":
            #     vg_embeddings[key] = vocab_glove["beside"]  # or unknown
            #     continue
            # elif key == "parked on":
            #     vg_embeddings[key] = vocab_glove["parked"]  # or unknown
            #     continue
            # elif key == "sitting in":
            #     vg_embeddings[key] = vocab_glove["sitting"]  # or unknown
            #     continue
            # vg_embeddings[key] = vocab_glove[key]

        embedding_weights = []

        #for i in range(len(vocab[type + "_name_to_idx"].items())):
        for i in range(len(vocab[type + "_idx_to_name"])):
            emb = vg_embeddings[vocab[type + "_idx_to_name"][i]]
            embedding_weights.append(emb)

        obj_embedding_weights = np.stack(embedding_weights)

        print("before size: ", num_instance, "after size: ", obj_embedding_weights.shape,)

        obj_embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(obj_embedding_weights).float())
        obj_embedding.weight.requires_grad = False

        return obj_embedding #, pred_embedding


    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

  # def forward(self, objs, triples, obj_to_img=None, boxes_gt=None, masks_gt=None, src_image=None, imgs_src=None,
  #             keep_box_idx=None, keep_feat_idx=None, keep_image_idx=None, combine_gt_pred_box_idx=None,
  #             query_feats=None, mode='train', t=0, query_idx=0, random_feats=False, get_layout_boxes=False,
  #             hide_obj_mask=None):
    def forward(self, objs, triples, obj_to_img=None, boxes_gt=None, src_image=None, imgs_src=None, masks_gt=None,
                keep_box_idx=None, keep_feat_idx=None, keep_image_idx=None, combine_gt_pred_box_idx=None,
                query_feats=None, mode='train', query_idx=0,
                t=0,  # ignored
                hide_obj_mask=None  # ignored, only for compatibility
                ):
        """
        Required Inputs:
        - objs: LongTensor of shape (num_objs,) giving categories for all objects
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Optional Inputs:
        - obj_to_img: LongTensor of shape (num_objs,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (num_objs, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        - src_image: (num_images, 3, H, W) input image to be modified
        - query_feats: feature vector from another image, to be used optionally in object replacement
        - keep_box_idx, keep_feat_idx, keep_image_idx: Tensors of ones or zeros, indicating
        what needs to be kept/masked on evaluation time.
        - combine_gt_pred_box_idx: Tensor of ones and zeros, indicating if size of pred box and position of gt boxes
          should be combined. Used in the "replace" mode.
        - mode: string, can take the option 'train' or one of the evaluation modes
        - t: iteration index, intended for debugging
        - query_idx: scalar id of object where query_feats should be used
        - random_feats: boolean. Used during evaluation to use noise instead of zeros for masked features phi
        - get_layout_boxes: boolean. If true, the boxes used for final layout construction are returned
        """

        assert mode in ["train", "eval", "auto_withfeats", "auto_nofeats", "reposition", "remove", "replace", "addition"]

        evaluating = mode != 'train'
        if evaluating:
            p_obj = self.p_mask_eval #0.3
        else:
            p_obj = self.p_obj

        in_image = src_image.clone()
        num_objs = objs.size(0)
        s, p, o = triples.chunk(3, dim=1) # All have shape (num_triples, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (num_triples,)


        edges = torch.stack([s, o], dim=1) #.to(self.device)  # Shape is (num_triples, 2)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs.clone()

        #mask graph
        #print(obj_vecs.shape)
        #obj_ones = torch.ones([num_objs, self.embedding_dim], dtype=obj_vecs.dtype, device=obj_vecs.device)
        obj_rand = torch.randn([num_objs, self.embedding_dim], dtype=obj_vecs.dtype, device=obj_vecs.device)
        mask_vec = torch.rand(num_objs, device=obj_vecs.device) < p_obj
        #print(mask_vec)
        mask_rep = mask_vec.repeat_interleave(self.embedding_dim).reshape(num_objs, self.embedding_dim)

        #obj_ones.scatter(obj_rand, mask_vec)
        obj_vecs = torch.where(mask_rep, obj_rand, obj_vecs)
        #obj_keep = F.dropout(obj_ones, self.p_obj) #* (1 - self.p_obj)
        #obj_vecs = obj_vecs * obj_keep

        if obj_to_img is None:
            obj_to_img = torch.zeros(num_objs, dtype=objs.dtype, device=objs.device)

        box_ones = torch.ones([num_objs, 1], dtype=boxes_gt.dtype, device=boxes_gt.device)
        box_keep, feats_keep = self.prepare_keep_idx(evaluating, box_ones, in_image.size(0), obj_to_img,
                                                     keep_box_idx, keep_feat_idx)

        boxes_prior = boxes_gt * box_keep

        if self.feats_in_gcn:
            obj_crop = get_cropped_objs(in_image, boxes_gt, obj_to_img, feats_keep, box_keep, evaluating, mode)

            high_feats = self.high_level_feats(obj_crop).to(self.device)

            high_feats = high_feats.view(high_feats.size(0), -1)
            high_feats = self.high_level_feats_fc(high_feats).to(self.device)

            feats_prior = high_feats * feats_keep

            #if evaluating and random_feats:
            #    # fill with noise the high level visual features, if the feature is masked/dropped
            #    normal_dist = tdist.Normal(loc=get_mean(self.spade_blocks), scale=get_std(self.spade_blocks))
            #    highlevel_noise = normal_dist.sample([high_feats.shape[0]])
            #    if not self.is_disentangled:
            #        feats_prior = feats_prior + (highlevel_noise.cuda() * (1 - feats_keep))

            # when a query image is used to generate an object of the same category
            if query_feats is not None:
                query_feats = query_feats.to(self.device)
                feats_prior[query_idx] = query_feats

        #if self.feats_in_gcn:
            obj_vecs = torch.cat([obj_vecs, boxes_prior, feats_prior], dim=1)

        else:
            obj_vecs = torch.cat([obj_vecs, boxes_prior], dim=1)

        obj_vecs = self.layer_norm(obj_vecs)

        pred_vecs = self.pred_embeddings(p).to(self.device)
        pred_vecs_orig = pred_vecs.clone()

        # GCN pass
        if self.gcn_type == "GCN":
            if isinstance(self.gconv, nn.Linear):
                obj_vecs = self.gconv(obj_vecs).to(self.device)
            else:
                obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)

        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        #print(obj_vecs.shape, pred_vecs.shape)
        if self.gconv_decoder is not None and not self.train_sg2im:
            obj_vecs, pred_vecs = self.gconv_decoder(obj_vecs, pred_vecs, edges)

        if edges.max() >= obj_vecs.shape[0] or edges.max() >= pred_vecs.shape[0]:
            print("we are doomed", edges.max(), obj_vecs.shape, pred_vecs.shape, obj_vecs_orig.shape, pred_vecs_orig.shape)
            assert False


        if self.train_sg2im:
            boxes_pred = self.box_net(obj_vecs)

            masks_pred = None
            #if self.mask_net is not None:
            #    mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
            #    masks_pred = mask_scores.squeeze(1).sigmoid()

            s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
            s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
            rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
            rel_scores = self.rel_aux_net(rel_aux_input)

            H, W = self.image_size
            layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

            #if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
            #else:
            #    layout_masks = masks_pred if masks_gt is None else masks_gt
            #    layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
            #                             obj_to_img, H, W)

            if self.layout_noise_dim > 0:
                N, C, H, W = layout.size()
                noise_shape = (N, self.layout_noise_dim, H, W)
                layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                           device=layout.device)
                layout = torch.cat([layout, layout_noise], dim=1)
            img = self.decoder_net(layout)
            return img, boxes_pred, masks_pred, rel_scores
        else:
            nodes = self.node_classifier(obj_vecs)
            rels = self.rel_classifier(pred_vecs)
            return obj_vecs, pred_vecs, obj_vecs_orig, pred_vecs_orig, edges, nodes, rels, p, mask_vec
          # return graph_features, num_objs, classification_scores, layout_boxes

    def bbox_coordinates_with_margin(self, bbox, margin, img):
        # extract bounding box with a margin

        left = max(0, bbox[0] * img.shape[2] - margin)
        top = max(0, bbox[1] * img.shape[1] - margin)
        right = min(img.shape[2], bbox[2] * img.shape[2] + margin)
        bottom = min(img.shape[1], bbox[3] * img.shape[1] + margin)

        return int(left), int(right), int(top), int(bottom)

    def forward_visual_feats(self, img, boxes):
        """
        gets VGG visual features from an image and box
        used for image query on evaluation time (evaluate_changes_vg.py)
        - img: Tensor of size [1, 3, H, W]
        - boxes: Tensor of size [4]
        return: feature vector in the RoI
        """

        left, right, top, bottom = get_left_right_top_bottom(boxes, img.size(2), img.size(3))

        obj_crop = img[0:1, :3, top:bottom, left:right]
        obj_crop = F.interpolate(obj_crop, size=(img.size(2) // 4, img.size(3) // 4), mode='bilinear', align_corners=True) #was upsample

        feats = self.high_level_feats(obj_crop)

        feats = feats.view(feats.size(0), -1)
        feats = self.high_level_feats_fc(feats)

        return feats

    def prepare_keep_idx(self, evaluating, box_ones, num_images, obj_to_img, keep_box_idx,
                         keep_feat_idx, with_feats=True):
        # random drop of boxes and visual feats on training time
        # use objs idx passed as argument on eval time
        imgbox_idx = torch.zeros(num_images, dtype=torch.int64, device=box_ones.device)
        for i in range(num_images):
            imgbox_idx[i] = (obj_to_img == i).nonzero()[-1]

        if evaluating:
            if keep_box_idx is not None:
                box_keep = keep_box_idx
            else:
                box_keep = box_ones

            if with_feats:
                if keep_feat_idx is not None:
                    feats_keep = keep_feat_idx
                else:
                    feats_keep = box_ones
        else:
            # drop random box(es) and feature(s)
            box_keep = F.dropout(box_ones, self.p_box, True, False) * (1 - self.p_box)
            if with_feats:
                feats_keep = F.dropout(box_ones, self.p, True, False) * (1 - self.p)

        # image obj cannot be dropped
        box_keep[imgbox_idx, :] = 1

        if with_feats:
            # image obj feats should not be dropped
            feats_keep[imgbox_idx, :] = 1
            return box_keep, feats_keep

        else:
            return box_keep


def get_left_right_top_bottom(box, height, width):
    """
    - box: Tensor of size [4]
    - height: scalar, image hight
    - width: scalar, image width
    return: left, right, top, bottom in image coordinates
    """
    left = (box[0] * width).type(torch.int32)
    right = (box[2] * width).type(torch.int32)
    top = (box[1] * height).type(torch.int32)
    bottom = (box[3] * height).type(torch.int32)

    return left, right, top, bottom


def mask_image_in_bbox(image, boxes, idx, obj_to_img, mode="normal"):
    """
    - image: Tensor of size [num_images, 4, H, W]
    - boxes: Tensor of size [num_objs, 4]
    - idx: scalar, object id
    - obj_to_img: Tensor of size [num_objs]
    - mode: string, "removal" if evaluating on removal mode, "normal" otherwise
    return: image, mask channel is set to ones in the bbox area of the object with id=idx
    """
    left, right, top, bottom = \
        get_left_right_top_bottom(boxes[idx], image.size(2), image.size(3))

    image[obj_to_img[idx], 3, top:bottom, left:right] = 1

    # on removal mode, make image area gray, so that features of removed object are not extracted
    if mode == "removal":
        image[obj_to_img[idx], :3, :, :] = put_gray_mask(image[obj_to_img[idx], :3, :, :],
                                                         image[obj_to_img[idx], 3:, :, :])

    return image


def put_gray_mask(imgs, mask):
    """
    fill image with gray pixels wherever mask is one
    - imgs: Tensor of size [num_images, 3, H, W] or [3, H, W]
    - mask: Tensor of size [num_images, 1, H, W] or [3, H, W]
    return: masked image of the same size as input image
    """

    reset = False
    if imgs.dim() == 3:
        imgs = torch.unsqueeze(imgs, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        reset = True

    imgs_masked = imgs * (1 - mask) + 0.5 * mask

    if reset:
        return imgs_masked[0]
    else:
        return imgs_masked


def get_cropped_objs(imgs, boxes, obj_to_img, feats_keeps, boxes_keeps, evaluating, mode, masked_feats=True):
    """
    prepare object RoIs for feature extraction
    - imgs: Tensor of size [num_images, 4, H, W]
    - boxes: Tensor of size [num_objs, 4]
    - obj_to_img: Tensor of size [num_objs]
    - feats_keeps: Tensor of size [num_objs]
    - boxes_keeps: Tensor of size [num_objs]
    - evaluating: boolean
    - mode: string, evaluation mode
    - masked_feats: boolean, if true image areas corresponding to dropped features are filled with gray pixels
    return: object RoI images, ready for feature extraction [num_objects, 3, H/4, W/4]
    """
    cropped_objs = []

    if masked_feats:
        features_mask = imgs[:, 3:4, :, :].clone()  # zeros

        if not evaluating:
            assert (imgs[:, 3, :, :].sum() == 0)

        for i in range(obj_to_img.size(0)):

            # during training: half the times, if the features are dropped then remove a patch from the image
            # during eval modes: drop objects that are going to change position
            if (feats_keeps[i, 0] == 0 and (i % 2 == 0 or evaluating)) or \
                    (boxes_keeps[i, 0] == 0 and evaluating):

                left, right, top, bottom = \
                    get_left_right_top_bottom(boxes[i], imgs.size(2), imgs.size(3))
                features_mask[obj_to_img[i], :, top:bottom, left:right] = 1

        # put gray as a mask over the image when things are dropped
        imgs_masked = put_gray_mask(imgs[:, :3, :, :], features_mask)

    for i in range(obj_to_img.size(0)):
        left, right, top, bottom = \
            get_left_right_top_bottom(boxes[i], imgs.size(2), imgs.size(3))

        # crop all objects
        try:
            if masked_feats and not (feats_keeps[i, 0] == 1 and boxes_keeps[i, 0] == 0 and evaluating):
                obj = imgs_masked[obj_to_img[i]:obj_to_img[i] + 1, :3, top:bottom, left:right]
            else:
                obj = imgs[obj_to_img[i]:obj_to_img[i] + 1, :3, top:bottom, left:right]
            obj = F.interpolate(obj, size=(imgs.size(2) // 4, imgs.size(3) // 4), mode='bilinear', align_corners=True) #upsample
        except:
            # cropped object has H or W zero
            obj = torch.zeros([1, imgs.size(1) - 1, imgs.size(2) // 4, imgs.size(3) // 4],
                              dtype=imgs.dtype, device=imgs.device)

        cropped_objs.append(obj)

    cropped_objs = torch.cat(cropped_objs, 0)

    return cropped_objs


def combine_boxes(gt, pred):
    """
    take position of gt bbox given as [left, top, right, bottom] in normalized coords
    and size of predicted bbox given as [left, top, right, bottom] in normalized coords
    used in object replacement, to adapt to the new class while keeping the original position
    """

    # center of gt bbox
    c_x = (gt[2] + gt[0]) / 2
    c_y = (gt[3] + gt[1]) / 2

    # pred H and W
    h = pred[3] - pred[1]
    w = pred[2] - pred[0]

    new_box = torch.zeros_like(gt)

    # update the comnined box, with a bit of care for corner cases
    new_box[0] = max(0.0, c_x - w / 2) # left
    new_box[2] = min(1.0, c_x + w / 2) # right
    new_box[1] = max(0.0, c_y - h / 2) # top
    new_box[3] = min(1.0, c_y + h / 2) # bottom

    return new_box
