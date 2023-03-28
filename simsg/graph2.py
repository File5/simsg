#!/usr/bin/python
#
# Copyright 2018 Google LLC
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

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn

import dgl
import dgl.function as dgl_fn
from dgl.nn.pytorch import edge_softmax, GraphConv
"""
PyTorch modules for dealing with graphs.
"""

def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer or final_nonlinearity:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  return nn.Sequential(*layers)


class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none', device=None):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim_obj
    self.input_dim_obj = input_dim_obj
    self.input_dim_pred = input_dim_pred
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling

    self.pooling = pooling
    net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)
    
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)


  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (num_objs, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (num_triples, D) giving vectors for all predicates
    - edges: LongTensor of shape (num_triples, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (num_objs, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (num_triples, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    #print(device, pred_vecs.device, edges.device)
    num_objs, num_triples = obj_vecs.size(0), pred_vecs.size(0)
    Din_obj, Din_pred, H, Dout = self.input_dim_obj, self.input_dim_pred, self.hidden_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (num_triples,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()
    #print(s_idx.device, o_idx.device)
    #print(obj_vecs.shape)
    #print(edges.max())

    # Get current vectors for subjects and objects; these have shape (num_triples, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]
    #print("1")
    # Get current vectors for triples; shape is (num_triples, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
    # p vecs have shape (num_triples, Dout)
    new_s_vecs = new_t_vecs[:, :H] #.cpu()
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)] #.cpu()
 
    # Allocate space for pooled object vectors of shape (num_objs, H)
    pooled_obj_vecs = torch.zeros(num_objs, H, dtype=dtype, device=device) #"cpu"

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (num_triples, D)
    #print("passed 1", pooled_obj_vecs.device, new_s_vecs.device) #

    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs) #.cpu()
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs) #.cpu()
    #print("passed 2")

    #print("passed 2", s_idx_exp.device, o_idx_exp.device, o_idx_exp.device, s_idx_exp.device)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)
    #print("passed 3")

    if self.pooling == 'avg':
      #print("here i am, would you send me an angel")
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(num_objs, dtype=dtype, device=device)
      ones = torch.ones(num_triples, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)
  
      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (num_objs, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)
    #print("I am here!")

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim_obj, input_dim_pred, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none', output_dim=None):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim_obj': input_dim_obj,
      'input_dim_pred': input_dim_pred,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization
    }
    for _ in range(self.num_layers-1):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    gconv_kwargs = {
      'input_dim_obj': input_dim_obj,
      'input_dim_pred': input_dim_pred,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
      'output_dim': output_dim
    }
    self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs



class GAT_layer(nn.Module):
    def __init__(self, input_dim_obj, input_dim_pred, use_obj_info, use_rel_info, k_update_steps, update_relations,
                 output_dim):
      super(GAT_layer, self).__init__()
      self.input_dim_obj = input_dim_obj
      self.input_dim_pred = input_dim_pred
      self.use_obj_info = use_obj_info
      self.use_rel_info = use_rel_info
      self.k_update_steps = k_update_steps
      self.update_relations = update_relations
      self.output_dim_obj = output_dim
      assert self.use_obj_info or self.use_rel_info

      self.linear_node = nn.Linear(self.input_dim_obj, self.input_dim_obj, bias=False)
      self.object_score = nn.Linear(self.input_dim_obj, 1, bias=False)
      self.linear_phi_node = nn.Linear(self.input_dim_obj * 2, self.input_dim_obj, bias=False)
      self.node_out_net = nn.Linear(self.input_dim_obj, self.output_dim_obj, bias=False)

      if self.use_rel_info or self.update_relations:
        self.linear_edge = nn.Linear(self.input_dim_pred, self.input_dim_obj, bias=False)
        self.relation_score = nn.Linear(self.input_dim_obj, 1, bias=False)
      if self.update_relations:
        self.linear_phi_edge = nn.Linear(self.input_dim_obj * 2, self.input_dim_pred, bias=False)
      self.edge_out_net = nn.Linear(self.input_dim_pred, self.output_dim_obj, bias=False)

      self.relu = nn.ReLU()

    def io_attention_send(self, edges):  #: dgl.EdgeBatch):
      # dict for storing messages to the nodes
      mail = dict()

      if self.use_rel_info or self.update_relations:
        s_e = self.relation_score(edges.data['F_e_t_new'])
        F_e = edges.data['F_e_t_new']
        if self.use_rel_info:
          mail['F_e'] = F_e
          mail['s_e'] = s_e

      if self.use_obj_info or self.update_relations:
        # Here edge.src is the data dict from the neighbour nodes
        s_n = edges.src['s_n']
        F_n = edges.src['F_n_t']
        if self.use_obj_info:
          mail['F_n'] = F_n
          mail['s_n'] = s_n

      if self.update_relations:
        # create and compute F_i and s_i, here edges.dst is the destination node or node_self/node_i
        F_i = edges.dst['F_n_t']
        s_i = edges.dst['s_n']
        # print(s_n.size(), s_i.size(), 'asdfasdf')
        s = torch.stack([s_n, s_i], dim=1)
        # print(s.size())
        F = torch.stack([F_n, F_i], dim=1)
        alpha_edge = torch.softmax(s, dim=1)
        applied_alpha = torch.sum(alpha_edge * F, dim=1)
        F_e_tplus1 = self.relu(self.linear_phi_edge(torch.cat([applied_alpha, F_e], dim=-1)))
        # mail['F_e_tplus1'] = F_e_tplus1
        edges.data['F_e_tplus1'] = F_e_tplus1
        # print(edges.data)
        # print("UPDATE EXECUTED")

      return mail

    def io_attention_reduce(self, nodes):  #: dgl.NodeBatch):
      # This is executed per node
      # print(nodes.mailbox['s_n'].size(), nodes.mailbox['s_e'].size())
      s_ne = torch.cat([nodes.mailbox['s_n'], nodes.mailbox['s_e']], dim=-2)
      # print(s_ne.size())
      # print(nodes.mailbox['F_n'].size(), nodes.mailbox['F_e'].size())
      F_ne = torch.cat([nodes.mailbox['F_n'], nodes.mailbox['F_e']], dim=-2)
      F_i = nodes.data['F_n_t']
      alpha_ne = torch.softmax(s_ne, dim=-2)
      applied_alpha = torch.sum(alpha_ne * F_ne, dim=-2)
      F_i_tplus1 = self.relu(self.linear_phi_node(torch.cat([applied_alpha, F_i], dim=-1)))
      # print(F_i_tplus1.size())
      # print('-----------------------------------------------------------------------')
      # print("REDUCED")
      return {'F_i_tplus1': F_i_tplus1}  # , 'F_e_tplus1': nodes.mailbox['F_e_tplus1']}

    def forward(self, obj_vecs, pred_vecs, edges):
      # print(self.input_dim_obj, self.input_dim_pred)
      # curr_t_vecs = self.extract_info(obj_vecs, pred_vecs, edges)
      g = dgl.DGLGraph()
      g = g.to("cuda:0")  # TODO

      g.add_nodes(obj_vecs.shape[0])
      g.ndata['F_n'] = obj_vecs

      for src, dst in edges:
        g.add_edges(src.item(), dst.item())

      g.edata['F_e'] = pred_vecs
      # print(g.ndata['F_n'].size(), g.edata['F_e'].size(), edges.size(), obj_vecs.size())

      if g.number_of_edges() > 0:
        g.ndata['F_n_t'] = g.ndata['F_n']
        g.edata['F_e_t'] = g.edata['F_e']

        for _ in range(self.k_update_steps):
          # print('loop')
          g.ndata['F_n_t'] = self.linear_node(g.ndata['F_n_t'])
          g.edata['F_e_t_new'] = self.linear_edge(g.edata['F_e_t'])

          g.ndata['s_n'] = self.object_score(g.ndata['F_n_t'])
          # print('s_n', g.ndata['s_n'].size())
          # g.send(message_func=self.io_attention_send, edges = g.edges)
          # g.recv(reduce_func=self.io_attention_reduce)
          g.update_all(message_func=self.io_attention_send, reduce_func=self.io_attention_reduce)
          # print("looped")
          # print('asdasd')
          # print("post", g.edges.data)
          g.ndata['F_n_t'] = g.ndata['F_i_tplus1']
          if self.update_relations:
            g.edata['F_e_t'] = g.edata['F_e_tplus1']
            # print(g.edata['F_e_tplus1'])

        # io = torch.split(g.ndata['F_n_t'], split_size_or_sections=b_num_nodes)
        obj_vecs, pred_vecs = g.ndata['F_n_t'], g.edata['F_e_t']
        obj_vecs = self.node_out_net(obj_vecs)
        pred_vecs = self.edge_out_net(pred_vecs)
      # else:
      #  io = torch.split(g.ndata['F_n'], split_size_or_sections=b_num_nodes)
      #
      # io = pad_sequence(io, batch_first=True)
      # io_mask = io.sum(dim=-1) != 0

      # return io, io_mask
      return obj_vecs, pred_vecs


class GAT(nn.Module):
  """ A sequence of scene graph convolution layers  """

  def __init__(self, args):
    super(GAT, self).__init__()

    self.num_layers = len(args.layers)
    self.gat_layers = nn.ModuleList()
    print("GAT LAYERS: ", args.layers)

    self.gconv_kwargs = {
      'input_dim_obj': args.input_dim_obj,
      'input_dim_pred': args.input_dim_pred,
      'output_dim': args.output_dim,
      'hidden_dim': args.hidden_dim,
      'pooling': args.gconv_pooling,
      'mlp_normalization': args.mlp_normalization,
    }

    self.gat_kwargs = {
      'input_dim_obj': args.input_dim_obj,
      'input_dim_pred': args.input_dim_pred,
      'use_obj_info': args.use_obj_info,
      'use_rel_info': args.use_rel_info,
      'k_update_steps': args.k_update_steps,
      'update_relations': args.update_relations,
      'output_dim': args.output_dim
    }

    for i, l in enumerate(args.layers):
      if l == "gat":
        if i != 0:
          self.gat_kwargs['input_dim_obj'] = args.output_dim
          self.gat_kwargs['input_dim_pred'] = args.output_dim
        '''if i == 0:
          self.gat_kwargs['output_dim'] = args.hidden_dim
        elif i != self.num_layers - 1:
          self.gat_kwargs['input_dim_obj'] = args.output_dim
          #self.gat_kwargs['output_dim'] = args.hidden_dim
        elif i == self.num_layers - 1:
          self.gat_kwargs['input_dim_obj'] = args.output_dim
          #self.gat_kwargs['output_dim'] = args.output_dim'''

        self.gat_layers.append(GAT_layer(**self.gat_kwargs))

      elif l == "gcn":
        if i != 0:
          self.gconv_kwargs['input_dim_obj'] = args.output_dim
          self.gconv_kwargs['input_dim_pred'] = args.output_dim
        '''if i = 0:
          #self.gconv_kwargs['output_dim'] = args.hidden_dim
        elif i != self.num_layers - 1:
          self.gconv_kwargs['input_dim_obj'] = args.hidden_dim
          self.gconv_kwargs['output_dim'] = args.hidden_dim
        elif i == self.num_layers - 1:
          self.gconv_kwargs['input_dim_obj'] = args.hidden_dim
          self.gconv_kwargs['output_dim'] = args.output_dim'''

        self.gat_layers.append(GraphTripleConv(**self.gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    # print('INPUT (OBJ, PRED): ', obj_vecs.size(), pred_vecs.size())
    for i in range(self.num_layers):
      layer = self.gat_layers[i]
      obj_vecs, pred_vecs = layer(obj_vecs, pred_vecs, edges)
      # print('LAYER OUTPUT (OBJ, PRED): ', obj_vecs.size(), pred_vecs.size())
    return obj_vecs, pred_vecs



class DisenTripletGCN(nn.Module):
    def __init__(self, args):
      super(DisenTripletGCN, self).__init__()
      self.Din_obj = args.input_dim_obj
      self.Din_pred = args.input_dim_pred
      self.H = args.hidden_dim
      self.Dout = args.out_dim

      self.net1 = DisenGCN(2 * args.input_dim_obj + args.input_dim_pred, 2 * args.hidden_dim + args.out_dim, args, split_mlp=False)
      self.net2 = DisenGCN(args.hidden_dim, args.out_dim, args, split_mlp=False)

    def extract_info(self, obj_vecs, pred_vecs, edges):
      self.dtype, self.device = obj_vecs.dtype, obj_vecs.device
      self.O, self.T = obj_vecs.size(0), pred_vecs.size(0)
      # Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

      # Break apart indices for subjects and objects; these have shape (T,)
      self.s_idx = edges[:, 0].contiguous()
      self.o_idx = edges[:, 1].contiguous()

      # Get current vectors for subjects and objects; these have shape (T, Din)
      cur_s_vecs = obj_vecs[self.s_idx]
      cur_o_vecs = obj_vecs[self.o_idx]

      # Get current vectors for triples; shape is (T, 3 * Din)
      # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
      cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
      return cur_t_vecs

    def pool_objs(self, out_feats):
      # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
      # p vecs have shape (T, Dout)
      dtype, device = self.dtype, self.device
      new_s_vecs = out_feats[:, :self.H]
      pred_vecs = out_feats[:, self.H:(self.H + self.Dout)]
      new_o_vecs = out_feats[:, (self.H + self.Dout):(2 * self.H + self.Dout)]

      # Allocate space for pooled object vectors of shape (O, H)
      pooled_obj_vecs = torch.zeros(self.O, self.H, dtype=dtype, device=device)

      # Use scatter_add to sum vectors for objects that appear in multiple triples;
      # we first need to expand the indices to have shape (T, D)
      s_idx_exp = self.s_idx.view(-1, 1).expand_as(new_s_vecs)
      o_idx_exp = self.o_idx.view(-1, 1).expand_as(new_o_vecs)
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(self.O, dtype=dtype, device=device)
      ones = torch.ones(self.T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, self.s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, self.o_idx, ones)

      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

      return pooled_obj_vecs, pred_vecs

    def forward(self, obj_vecs, pred_vecs, edges):
      curr_t_vecs = self.extract_info(obj_vecs, pred_vecs, edges)
      out_feats = self.net1(curr_t_vecs, edges)

      so_vecs, pred_vecs = self.pool_objs(out_feats)
      obj_vecs = self.net2(so_vecs, edges)

      return obj_vecs, pred_vecs


class NeibRoutLayer(nn.Module):
  def __init__(self, num_caps, niter, tau=1.0):
    super(NeibRoutLayer, self).__init__()
    self.k = num_caps
    self.niter = niter
    self.tau = tau

  #
  # x \in R^{n \times d}: d-dimensional node representations.
  #    It can be node features, output of the previous layer,
  #    or even rows of the adjacency matrix.
  #
  # src_trg \in R^{2 \times m}: a list that contains m edges.
  #    src means the source nodes of the edges, and
  #    trg means the target nodes of the edges.
  #
  def forward(self, x, src_trg):
    m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
    n, d = x.shape
    k, delta_d = self.k, d // self.k
    x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
    z = x[src].view(m, k, delta_d)
    u = x
    scatter_idx = trg.view(m, 1).expand(m, d)
    for clus_iter in range(self.niter):
      p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
      p = fn.softmax(p / self.tau, dim=1)
      scatter_src = (z * p.view(m, k, 1)).view(m, d)
      u = torch.zeros(n, d, device=x.device)
      u.scatter_add_(0, scatter_idx, scatter_src)
      u += x
      # noinspection PyArgumentList
      u = fn.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
    # p: m * k, m is #edges
    return u  # , p


class DisenGCN(nn.Module):
  #
  # nfeat: dimension of a node's input feature
  # nclass: the number of target classes
  # hyperpm: the hyper-parameter configuration
  #    ncaps: the number of capsules/channels/factors per layer
  #    routit: routing iterations
  #
  def __init__(self, nfeat, nclass, hyperpm, split_mlp=False):
    super(DisenGCN, self).__init__()
    self.input_dim = nfeat
    self.output_dim = nclass
    self.pca = SparseInputLinear(nfeat, hyperpm.ncaps * hyperpm.nhidden)
    self.bn = nn.BatchNorm1d(nclass)
    conv_ls = []
    for i in range(hyperpm.nlayer):
      conv = NeibRoutLayer(hyperpm.ncaps, hyperpm.routit)
      self.add_module('conv_%d' % i, conv)
      conv_ls.append(conv)
    self.conv_ls = conv_ls
    if split_mlp:
      self.clf = SplitMLP(nclass, hyperpm.nhidden * hyperpm.ncaps,
                          nclass)
    else:
      self.clf = nn.Linear(hyperpm.nhidden * hyperpm.ncaps, nclass)
    self.dropout = hyperpm.dropout

  def _dropout(self, x):
    return fn.dropout(x, self.dropout, training=self.training)

  def forward(self, obj_vec, edges):
    obj_vec = self.pca(obj_vec)
    obj_vec = self._dropout(fn.leaky_relu(obj_vec))
    for conv in self.conv_ls:
      obj_vec = conv(obj_vec, edges)
      # obj_vec = self._dropout(fn.leaky_relu(obj_vec))
    obj_vec = self.clf(obj_vec)
    obj_vec = self.bn(obj_vec)
    obj_vec = fn.leaky_relu(obj_vec)
    return obj_vec



class SparseInputLinear(nn.Module):
  def __init__(self, inp_dim, out_dim):
    super(SparseInputLinear, self).__init__()
    weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
    weight = nn.Parameter(torch.from_numpy(weight))
    bias = np.zeros(out_dim, dtype=np.float32)
    bias = nn.Parameter(torch.from_numpy(bias))
    self.inp_dim, self.out_dim = inp_dim, out_dim
    self.weight, self.bias = weight, bias
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / np.sqrt(self.weight.shape[1])
    self.weight.data.uniform_(-stdv, stdv)
    self.bias.data.uniform_(-stdv, stdv)

  def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
    return torch.mm(x, self.weight) + self.bias


class SplitMLP(nn.Module):
  def __init__(self, n_mlp, d_inp, d_out):
    super(SplitMLP, self).__init__()
    assert d_inp >= n_mlp and d_inp % n_mlp == 0
    assert d_out >= n_mlp and d_out % n_mlp == 0
    self.mlps = nn.Conv1d(in_channels=n_mlp, out_channels=d_out,
                          kernel_size=d_inp // n_mlp, groups=n_mlp)
    self.n_mlp = n_mlp

  def forward(self, x):
    n = x.shape[0]
    x = x.view(n, self.n_mlp, -1)
    x = self.mlps(x)
    x = x.view(n, -1)
    return x

class FactorTripletGCN(nn.Module):
    def __init__(self, args):
      super(FactorTripletGCN, self).__init__()
      self.Din_obj = args.input_dim_obj
      self.Din_pred = args.input_dim_pred
      self.H = args.hidden_dim
      self.Dout = args.out_dim

      # feat = torch.ones(1, in_dim)

      # g, features, train_mask, val_mask, test_mask, factor_graph = (None, feat, None, None, None, None)
      g = None

      # num_feats = features.shape[1]
      # n_classes = out_dim

      self.net1 = FactorGNN(g,
                            args.num_layers,
                            2 * args.input_dim_obj + args.input_dim_pred,  # num_feats,
                            args.num_hidden,
                            2 * args.hidden_dim + args.out_dim,  # n_classes,
                            args.num_latent,
                            args.in_drop,
                            args.residual)

      self.net2 = FactorGNN(g,
                            args.num_layers,
                            args.hidden_dim,  # num_feats,
                            args.num_hidden,
                            args.out_dim,  # n_classes,
                            args.num_latent,
                            args.in_drop,
                            args.residual)

      # self.net1 = FactorGNN(3 * in_dim, 2 * hidden_dim + out_dim, hyperpm)
      # self.net2 = FactorGNN(hidden_dim, out_dim, hyperpm)

    def compute_disentangle_loss(self):
      # compute disentangle loss at each layer
      # return: list of loss
      net1_loss = self.net1.compute_disentangle_loss()
      net1_loss = self.net1.merge_loss(net1_loss)
      net2_loss = self.net2.compute_disentangle_loss()
      net2_loss = self.net2.merge_loss(net2_loss)
      loss_list = net1_loss + net2_loss
      # loss_list = [net1_loss, net2_loss]
      # loss_list = torch.as_tensor(loss_list)
      return loss_list

    def extract_info(self, obj_vecs, pred_vecs, edges):
      self.dtype, self.device = obj_vecs.dtype, obj_vecs.device
      self.O, self.T = obj_vecs.size(0), pred_vecs.size(0)
      # Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

      # Break apart indices for subjects and objects; these have shape (T,)
      self.s_idx = edges[:, 0].contiguous()
      self.o_idx = edges[:, 1].contiguous()

      # Get current vectors for subjects and objects; these have shape (T, Din)
      cur_s_vecs = obj_vecs[self.s_idx]
      cur_o_vecs = obj_vecs[self.o_idx]

      # Get current vectors for triples; shape is (T, 3 * Din)
      # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
      cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
      return cur_t_vecs

    def pool_objs(self, out_feats):
      # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
      # p vecs have shape (T, Dout)
      dtype, device = self.dtype, self.device
      new_s_vecs = out_feats[:, :self.H]
      pred_vecs = out_feats[:, self.H:(self.H + self.Dout)]
      new_o_vecs = out_feats[:, (self.H + self.Dout):(2 * self.H + self.Dout)]

      # Allocate space for pooled object vectors of shape (O, H)
      pooled_obj_vecs = torch.zeros(self.O, self.H, dtype=dtype, device=device)

      # Use scatter_add to sum vectors for objects that appear in multiple triples;
      # we first need to expand the indices to have shape (T, D)
      s_idx_exp = self.s_idx.view(-1, 1).expand_as(new_s_vecs)
      o_idx_exp = self.o_idx.view(-1, 1).expand_as(new_o_vecs)
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(self.O, dtype=dtype, device=device)
      ones = torch.ones(self.T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, self.s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, self.o_idx, ones)

      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

      return pooled_obj_vecs, pred_vecs

    def forward(self, obj_vecs, pred_vecs, edges):
      curr_t_vecs = self.extract_info(obj_vecs, pred_vecs, edges)
      g = dgl.DGLGraph()
      g = g.to("cuda:0")
      g.add_nodes(curr_t_vecs.shape[0])
      g.ndata['feat'] = curr_t_vecs

      for src, dst in edges:
        g.add_edges(src.item(), dst.item())

      # print("cat vecs: ", curr_t_vecs.shape)
      self.net1.g = g
      out_feats = self.net1(curr_t_vecs)
      # print("net 1 out: ", out_feats.shape)

      so_vecs, pred_vecs = self.pool_objs(out_feats)
      g2 = dgl.DGLGraph()
      g2 = g2.to("cuda:0")
      g2.add_nodes(so_vecs.shape[0])
      g2.ndata['feat'] = so_vecs

      for src, dst in edges:
        g2.add_edges(src.item(), dst.item())
      self.net2.g = g2
      obj_vecs = self.net2(so_vecs)

      return obj_vecs, pred_vecs


class FactorGNN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_latent,
                 feat_drop,
                 residual):
      super(FactorGNN, self).__init__()
      self.g = g
      self.layers = nn.ModuleList()
      self.BNs = nn.ModuleList()
      self.linears = nn.ModuleList()
      self.feat_drop = feat_drop

      self.linears.append(nn.Linear(in_dim, num_classes))

      self.layers.append(DisentangleLayer(num_latent, in_dim, num_hidden, cat=True))
      self.BNs.append(nn.BatchNorm1d(num_hidden))
      self.linears.append(nn.Linear(num_hidden, num_classes))

      self.layers.append(DisentangleLayer(num_latent, num_hidden, num_hidden, cat=True))
      self.BNs.append(nn.BatchNorm1d(num_hidden))
      self.linears.append(nn.Linear(num_hidden, num_classes))

      self.layers.append(DisentangleLayer(max(num_latent // 2, 1), num_hidden, num_hidden, cat=True))
      self.BNs.append(nn.BatchNorm1d(num_hidden))
      self.linears.append(nn.Linear(num_hidden, num_classes))

      self.layers.append(DisentangleLayer(max(num_latent // 2, 1), num_hidden, num_hidden, cat=True))
      self.BNs.append(nn.BatchNorm1d(num_hidden))
      self.linears.append(nn.Linear(num_hidden, num_classes))

      self.layers.append(DisentangleLayer(max(num_latent // 2 // 2, 1), num_hidden, num_hidden, cat=True))
      self.BNs.append(nn.BatchNorm1d(num_hidden))
      self.linears.append(nn.Linear(num_hidden, num_classes))

      self.activate = torch.nn.ReLU()

    def forward(self, inputs):
      self.feat_list = []

      feat = inputs
      self.feat_list.append(feat)
      for layer, bn in zip(self.layers, self.BNs):
        # feat = torch_fn.dropout(feat, self.feat_drop)
        pre_feat = feat
        feat = layer(self.g, feat)
        feat = bn(feat)
        # feat = feat + pre_feat
        feat = self.activate(feat)

        self.feat_list.append(feat)

      logit = 0
      for feat, linear in zip(self.feat_list, self.linears):
        self.g.ndata['h'] = feat
        # print("before feat 2: ", feat.shape)
        # self.g = dgl.batch(self.g)
        h = feat
        # print("befaft feat 2: ", h.shape)
        # h = self.activate(h)

        h = linear(h)
        # print("after feat 2: ", h.shape)
        logit += fn.dropout(h, self.feat_drop)

      return logit

    def get_hidden_feature(self):
      return self.feat_list

    def get_factor(self):
      # return factor graph at each disentangle layer as list
      factor_list = []
      for layer in self.layers:
        if isinstance(layer, DisentangleLayer):
          factor_list.append(layer.get_factor())
      return factor_list

    def compute_disentangle_loss(self):
      # compute disentangle loss at each layer
      # return: list of loss
      loss_list = []
      for layer in self.layers:
        if isinstance(layer, DisentangleLayer):
          loss_list.append(layer.compute_disentangle_loss())
      return loss_list

    @staticmethod
    def merge_loss(list_loss):
      total_loss = 0
      for loss in list_loss:
        discrimination_loss, distribution_loss = loss[0], loss[1]
        total_loss += discrimination_loss
        # total_loss += distribution_loss
      return total_loss


class DisentangleLayer(nn.Module):
  def __init__(self, n_latent, in_dim, out_dim, cat=True):
    super(DisentangleLayer, self).__init__()
    # init self.g as None, after forward step, it will be replaced
    self.g = None

    self.n_latent = n_latent
    self.n_feat_latent = out_dim // self.n_latent if cat else out_dim
    self.cat = cat

    self.linear = nn.Linear(in_dim, self.n_feat_latent)
    self.att_ls = nn.ModuleList()
    self.att_rs = nn.ModuleList()
    for latent_i in range(self.n_latent):
      self.att_ls.append(nn.Linear(self.n_feat_latent, 1))
      self.att_rs.append(nn.Linear(self.n_feat_latent, 1))

    # define for the additional losses
    self.graph_to_feat = GraphEncoder(self.n_feat_latent, self.n_feat_latent // 2)
    self.classifier = nn.Linear(self.n_feat_latent, self.n_latent)
    self.loss_fn = nn.BCEWithLogitsLoss()  # changed CrossEntropyLoss()

  def forward(self, g, inputs):
    self.g = g.local_var()
    out_feats = []
    hidden = self.linear(inputs)
    self.hidden = hidden
    for latent_i in range(self.n_latent):
      # compute factor features of nodes
      a_l = self.att_ls[latent_i](hidden)
      a_r = self.att_rs[latent_i](hidden)
      self.g.ndata.update({f'feat_{latent_i}': hidden,
                           f'a_l_{latent_i}': a_l,
                           f'a_r_{latent_i}': a_r})
      self.g.apply_edges(dgl_fn.u_add_v(f'a_l_{latent_i}', f'a_r_{latent_i}', f"factor_{latent_i}"))
      self.g.edata[f"factor_{latent_i}"] = torch.sigmoid(6.0 * self.g.edata[f"factor_{latent_i}"])
      feat = self.g.ndata[f'feat_{latent_i}']

      # graph conv on the factor graph
      norm = torch.pow(self.g.in_degrees().float().clamp(min=1), -0.5)
      shp = norm.shape + (1,) * (feat.dim() - 1)
      norm = torch.reshape(norm, shp).to(feat.device)
      feat = feat * norm

      # generate the output features
      self.g.ndata['h'] = feat
      self.g.update_all(dgl_fn.u_mul_e('h', f"factor_{latent_i}", 'm'),
                        dgl_fn.sum(msg='m', out='h'))
      out_feats.append(self.g.ndata['h'].unsqueeze(-1))

    if self.cat:
      return torch.cat(tuple([rst.squeeze(-1) for rst in out_feats]), -1)
    else:
      return torch.mean(torch.cat(tuple(out_feats), -1), -1)

  def compute_disentangle_loss(self):
    assert self.g is not None, "compute disentangle loss need to be called after forward pass"

    # compute discrimination loss
    factors_feat = [self.graph_to_feat(self.g, self.hidden, f"factor_{latent_i}").squeeze()
                    for latent_i in range(self.n_latent)]

    # labels = [torch.ones(f.shape[0]) * i for i, f in enumerate(factors_feat)]
    # labels = torch.cat(tuple(labels), 0).long().cuda()
    factors_feat = torch.cat(tuple(factors_feat), 0)
    factors_feat = factors_feat.reshape(-1, self.n_feat_latent)

    # labels = labels.reshape(-1, self.n_feat_latent)
    # print(factors_feat.shape, self.n_feat_latent, self.n_latent)

    pred = self.classifier(factors_feat)
    labels = torch.ones_like(pred)  # [torch.ones(f.shape[0]) * i for i, f in enumerate(factors_feat)]
    # labels = torch.cat(tuple(labels), 0).long().cuda()
    # pred = pred.reshape(1, -1)
    # print(pred.shape, labels.shape, factors_feat.shape)
    discrimination_loss = self.loss_fn(pred, labels)

    distribution_loss = 0

    return [discrimination_loss, distribution_loss]

  def get_factor(self):
    g = self.g.local_var()
    return g


class GraphEncoder(nn.Module):
  def __init__(self, in_dim, hidden_dim):
    super(GraphEncoder, self).__init__()
    self.linear1 = nn.Linear(in_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, in_dim)

  def forward(self, g, inputs, factor_key):
    g = g.local_var()
    # graph conv on the factor graph
    feat = self.linear1(inputs)
    norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)
    shp = norm.shape + (1,) * (feat.dim() - 1)
    norm = torch.reshape(norm, shp).to(feat.device)
    feat = feat * norm

    g.ndata['h'] = feat
    g.update_all(dgl_fn.u_mul_e('h', factor_key, 'm'),
                 dgl_fn.sum(msg='m', out='h'))
    g.ndata['h'] = torch.tanh(g.ndata['h'])

    # graph conv on the factor graph
    feat = self.linear2(g.ndata['h'])
    feat = feat * norm

    g.ndata['h'] = feat
    g.update_all(dgl_fn.u_mul_e('h', factor_key, 'm'),
                 dgl_fn.sum(msg='m', out='h'))
    g.ndata['h'] = torch.tanh(g.ndata['h'])

    h = dgl.mean_nodes(g, 'h').unsqueeze(-1)
    h = torch.tanh(h)
