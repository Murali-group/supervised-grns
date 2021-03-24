#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os.path as osp

import argparse

import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from tqdm import tqdm 
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torchvision

import math
import random

from itertools import combinations


from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected, negative_sampling
import networkx as nx



# In[3]:
import math
import random
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, precision_recall_curve, auc
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.nn.inits import reset

#https://stackoverflow.com/questions/31645314/custom-precision-at-k-scoring-object-in-sklearn-for-gridsearchcv
def precision_at_k(y_true, y_score, k):
    df = pd.DataFrame({'true': y_true.tolist(), 'score': y_score.tolist()}).sort_values('score', ascending=False)
    df.reset_index(inplace = True)
    threshold = df.loc[k-1]['score']
    df = df[df.score >= threshold]
    return df.true.sum()/df.shape[0]

def compute_metrics(actual, predicted):
    y_true = np.concatenate(actual)
    y_predicted = np.concatenate(predicted)
    compute_auc(y_true, y_predicted)

def compute_auc(y_true, y_predicted):
    precision, recall, _ = precision_recall_curve(y_true, y_predicted)
    auc_score = auc(recall, precision)
    print("Overall AUC=%.4f" %auc_score)
    return auc_score

def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()

    return opts


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape



class GAEwithK(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super(GAEwithK, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAEwithK.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)


    def recon_loss(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              1e-15).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)

        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        pr, rec, thresholds = precision_recall_curve(y, pred)
        auc_score = auc(rec, pr)
        #pd.DataFrame([y, pred], index = ['true','pred']).T.to_csv('preds.csv')
        #pd.DataFrame([pr, rec, thresholds], index = ['pr','rec','thres']).T.to_csv('pr.csv')
        return precision_at_k(y, pred, pos_edge_index.size(1)),average_precision_score(y, pred), auc_score, pred, y

    

class Encoder(torch.nn.Module):
    def __init__(self, h_sizes):
        super(Encoder, self).__init__()
        
        self.hidden = nn.ModuleList()
        self.num_hidden = len(h_sizes) - 1
        for k in range(self.num_hidden):
            self.hidden.append(GCNConv(h_sizes[k], h_sizes[k+1], cached=False))

    def forward(self, x, edge_index):
        for k in range(self.num_hidden):
            x = F.relu(self.hidden[k](x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        return x


class TFDecoder(torch.nn.Module):
    def __init__(self, num_nodes, TFIDs):
        super(TFDecoder, self).__init__()
        self.TFIDs = list(TFIDs)
        self.num_nodes = num_nodes
        self.in_dim = 1 # one relation type
        self.weight = nn.Parameter(torch.Tensor(self.num_nodes))
        self.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        zNew = torch.mul(z.t(),self.weight).t()
        value = (zNew[edge_index[0]]*z[edge_index[1]]).sum(dim=1)
        
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
         self.weight.data.normal_(std=1/np.sqrt(self.in_dim))

            
class RESCALDecoder(torch.nn.Module):
    def __init__(self, out_dim):
        super(RESCALDecoder, self).__init__()
        self.out_dim = out_dim
        self.in_dim = 1 # one relation type
        self.weight = nn.Parameter(torch.Tensor(self.out_dim,self.out_dim))
        self.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        zNew = torch.matmul(z.clone(),self.weight)
        value = (zNew[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
         self.weight.data.normal_(std=1/np.sqrt(self.in_dim))

