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
from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv, GMMConv, GATConv
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

from itertools import combinations, permutations, product


from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected, negative_sampling
import networkx as nx



# In[3]:
import math
import random
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, precision_recall_curve
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
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

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
        pd.DataFrame([y, pred], index = ['true','pred']).T.to_csv('preds.csv')
        pd.DataFrame([pr, rec, thresholds], index = ['pr','rec','thres']).T.to_csv('pr.csv')
        return precision_at_k(y, pred, pos_edge_index.size(1)),average_precision_score(y, pred)

    

class Encoder(torch.nn.Module):
    def __init__(self, h_sizes):
        super(Encoder, self).__init__()
        
        self.hidden = nn.ModuleList()
        self.num_hidden = len(h_sizes) - 1
        for k in range(self.num_hidden):
            self.hidden.append(GCNConv(h_sizes[k], h_sizes[k+1], cached=False))

    def forward(self, x, edge_index):
        for k in range(self.num_hidden):
            #x = F.relu(F.dropout(self.hidden[k](x, edge_index), p=0.2, training=self.training))
            x = F.relu(self.hidden[k](x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        return x


class TFDecoder(torch.nn.Module):
    def __init__(self, num_nodes, TFIDs):
        super(TFDecoder, self).__init__()
        self.TFIDs = list(TFIDs)
        self.num_nodes = num_nodes
        self.in_dim = 1 # one relation type
        #self.weight = nn.Parameter(torch.Tensor(len(self.TFIDs)))
        self.weight = nn.Parameter(torch.Tensor(self.num_nodes))
        self.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        #newWeight = torch.zeros(self.num_nodes).to(dev)
        #nCnt = 0
        #for idx in self.TFIDs:
        #    newWeight[idx] = self.weight[nCnt]
        #    nCnt += 1
        zNew = torch.mul(z.t(),self.weight).t()
        #print(self.weight[0])#, newWeight[850])
        #sys.exit()
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

        #zNew = z.clone()*self.weight
        zNew = torch.matmul(z.clone(),self.weight)
        #zNew = z*self.weight
        #print(edge_index)
        #print(zNew.shape,self.weight)
        value = (zNew[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        #self.weight[edge_index[0]]
        #print(value, edge_index)
        #value = (1 * z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        #print(edge_index.shape,value.shape)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
         self.weight.data.normal_(std=1/np.sqrt(self.in_dim))

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_only_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return (loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    epr, ap = model.test(z, pos_edge_index, neg_edge_index)
    return z, epr, ap

def val(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
    return loss



def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(description='Parse arguments.')

    parser.add_argument('-e','--expFile', 
                        default = '/home/malabika/GRN/hsc_combined1k.csv',
                        help='Path to expression data file.Required. \n')

    parser.add_argument('-n','--netFile', 
                        default = '/home/malabika/GRN/STRINGc700.csv',
                        help='Path to network file.Required. \n')

    parser.add_argument('-t','--test', type=int, default = 0.3,
                        help='Fraction of total edges to use for test. Default = 0.3. \n')

    parser.add_argument('-k','--kTrain', type=int, default = 1,
                        help='Ratio training negatives:positives. \
                        Example, use --kTrain=10 for 10 negatives for 1 positive.  Default = 1. \n')

    parser.add_argument('-s','--kTest', type=int, default = 1,
                        help='Ratio of test negatives:positives. \
                        Example, use --kTest=10 for 10 negatives for 1 positive. Default = 1. \n')
    
    parser.add_argument('-i','--ignoreComp', action='store_true', default = False,
                        help='Ignore computation of matrices needed for generating train and test splits. \n')
        
    parser.add_argument('-p','--epochs', type=int, default = 1000,
                        help='Number of epochs. Default = 1000. \n')

    parser.add_argument('-a','--type', type=str, default = 'E',
                        help='Type: E: Expression data, P: PCA, I: Adjacency matrix. Default = E. \n')
    
    parser.add_argument('-r','--rand', type=int, default=2019,
                        help='random seed ID. Default=2019')
    
    parser.add_argument('-d','--id', type=int, default=0,
                        help='ID of the TF to be removed for evaluation. Default=0')
    
    parser.add_argument('-l','--hidden', type=int, default=2,
                        help = 'Number of GCN layers. Default = 2')
        
    parser.add_argument('-y','--encoder', default='GCN',
                        help = 'Type of encoder: For directed graph encoder:\
                        DGCN, undirected graph encoder: GCN. Default=GCN')
                        
    parser.add_argument('-z','--decoder', default='IP',
                        help = 'Type of decoder: IP: Innerproduct decoder,\
                        TF: TFMult Decoder, RS: Rescale decoder.')
                        
    parser.add_argument('-g','--log', action='store_true', default = False,
                        help = 'Take log of RPKM expression values. Default=False.')
    return parser


EPS = 1e-15
MAX_LOGVAR = 10


# In[2]:
opts = parse_arguments()

import random
random.seed(opts.rand)
np.random.seed(opts.rand)
torch.manual_seed(opts.rand)


if opts.log:
    ExpDF = np.log10(pd.read_csv(opts.expFile, index_col = 0)+10**-4)
else:
    ExpDF = pd.read_csv(opts.expFile, index_col = 0)
GeneralChIP = pd.read_csv(opts.netFile)

#GeneralChIP.columns = ['Gene1','Gene2']
GeneralChIP.Gene1 = GeneralChIP.Gene1.str.upper()
GeneralChIP.Gene2 = GeneralChIP.Gene2.str.upper()
GeneralChIP = GeneralChIP[(GeneralChIP['Gene1']!= GeneralChIP['Gene2'])]

GeneralChIP = GeneralChIP[(GeneralChIP['Gene1'].isin(list(ExpDF.index))) & (GeneralChIP['Gene2'].isin(list(ExpDF.index)))]
GeneralChIP = GeneralChIP.reset_index(drop = True)


UnDirGr = nx.from_pandas_edgelist(GeneralChIP,source='Gene1', target='Gene2', create_using = nx.DiGraph)
newUnDirGr = nx.convert_node_labels_to_integers(UnDirGr,ordering = 'sorted', label_attribute = 'name')

NodeLst = sorted(list(newUnDirGr.nodes()))


onlyGenes = []
onlyTFs = []
TFNames = {}

for n,data in newUnDirGr.nodes(data=True):
    #print(n,data['name'])
    if data['name'] not in GeneralChIP.Gene1.values:
        onlyGenes.append(n)
    else:
        onlyTFs.append(n)
        TFNames[n] = data['name']
#print(len(onlyGenes),len(onlyTFs))

# Get a list of all possible edges
#possibleEdges = set(permutations(NodeLst, r = 2))
possibleEdges = set(product(onlyTFs, NodeLst))
#print(len(possibleEdges), len(NodeLst))


#cnt = 0
#for x in permutations(sorted(onlyGenes),2):
#    possibleEdges.remove(x)
    
#print(len(possibleEdges))

# This order of nodes in newUnDirGr is same as UnDirGr
NodeNames = sorted(list(UnDirGr.nodes()))
subDF = ExpDF.loc[NodeNames]
subDF = subDF.loc[~subDF.index.duplicated(keep='first')]
subDF = subDF.div(subDF.sum(axis=1), axis=0).fillna(0)


pCnt = 0
nCnt = 0
# Positive edge IDs
posE = np.zeros((len(UnDirGr.edges()),2))

# negative edge IDs
negE = np.zeros((len(possibleEdges)-len(UnDirGr.edges()),2))
#print(posX.shape, negX.shape, len(NodeLst))

if opts.ignoreComp:
    posE = np.load('posE.npy').astype(int)
    negE = np.load('negE.npy').astype(int)
else:
    for edge in tqdm(possibleEdges):
        if edge in newUnDirGr.edges():
            posE[pCnt] = edge
            pCnt += 1                
        else:
            negE[nCnt] = edge
            nCnt += 1
    np.save('posE',posE)
    np.save('negE',negE)
    sys.exit()



refNetDF = GeneralChIP
sourceNodes = [list(subDF.index.values).index(x) for x in refNetDF['Gene1'].values]

uniqSN =  np.unique(sourceNodes)
random.shuffle(uniqSN)
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, random_state=opts.rand, shuffle=False)
iCnt = 0
for train_index, test_index in cv.split(uniqSN):
    if iCnt == opts.id-1:
        sNodes=uniqSN[test_index]
    iCnt +=1
#print(sNodes)
#sNodes = [68]
#print(sNodes)
test_posIdx = np.array([],dtype = int)
#count positives for each edge
tfIDDict = {}
for tfID in sNodes:
    newArr = np.where(np.isin(posE,tfID))[0]
    test_posIdx = np.hstack((test_posIdx,newArr))
    tfIDDict[tfID] = len(newArr)
    
train_posIdx = np.setdiff1d(np.arange(0,len(posE)),test_posIdx)


all_test_negIdx = np.array([],dtype = int)
test_negIdx = np.array([],dtype = int)
for tfID in sNodes:
    newArr = np.where(np.isin(negE,tfID))[0]
    random.shuffle(newArr)
    nTest = min(opts.kTest*tfIDDict[tfID],len(newArr))
    all_test_negIdx = np.hstack((all_test_negIdx,newArr))
    test_negIdx = np.hstack((test_negIdx,newArr[:nTest]))

    
all_train_negIdx =np.setdiff1d(np.arange(0,len(negE)), all_test_negIdx)
nTrain = min(opts.kTrain*len(train_posIdx),len(all_train_negIdx))
train_negIdx = np.random.choice(all_train_negIdx,nTrain) 
#print(sNodes, negE[test_negIdx])

#print(len(train_posIdx), len(train_negIdx))
#print(set(test_posIdx).intersection(train_posIdx), set(test_negIdx).intersection(train_negIdx))

ExpDF = ExpDF.loc[NodeNames]
ExpDF = ExpDF.div(ExpDF.sum(axis=1), axis=0).fillna(0)

if opts.type == 'E':
    exprDF = ExpDF.copy()
elif opts.type == 'P':
    exprDF=pd.read_csv('BM-PCA-DF.csv', index_col=0, header=0)
    #print(exprDF.head())
    #sys.exit()
    #exprDF =  pd.DataFrame(PCA(n_components=0.75).fit_transform(ExpDF.values), index=ExpDF.index)
    #exprDF.to_csv('BM-PCA-DF.csv')
    #sys.exit()
elif opts.type == 'I':
    exprDF = pd.DataFrame(np.eye(ExpDF.shape[0],ExpDF.shape[0]),index = ExpDF.index, columns = ExpDF.index)
else:
    print("WRONG")

val_posIdx = random.sample(list(train_posIdx), int(0.1*len(train_posIdx)))
train_posIdx = list(set(train_posIdx).difference(set(val_posIdx)))

val_negIdx = random.sample(list(train_negIdx), int(0.1*len(train_negIdx)))
train_negIdx = list(set(train_negIdx).difference(set(val_negIdx)))

sourceNodes = posE[train_posIdx , 0]
targetNodes = posE[train_posIdx , 1]

sourceNodesCPY = posE[train_posIdx , 0]
targetNodesCPY = posE[train_posIdx , 1]


for tfID in tqdm(sNodes):
    sourceNodes = np.hstack((sourceNodes, [tfID]*len(NodeLst)))
    targetNodes = np.hstack((targetNodes, NodeLst))

nodeFeatures = torch.Tensor(exprDF.values)

if opts.encoder == 'GCN':
    eIndex = to_undirected(torch.LongTensor([sourceNodes, targetNodes]))
else:
    eIndex = torch.LongTensor([sourceNodes, targetNodes])
    
data = Data(x=nodeFeatures, edge_index=eIndex)

#print(eIndex, sorted(TFNames.keys()))
#sys.exit()

if opts.encoder == 'GCN':
    data.train_pos_edge_index = to_undirected(torch.stack([torch.LongTensor(sourceNodes), torch.LongTensor(targetNodes)], dim=0))
    #print(data.train_pos_edge_index.shape)
    data.train_pos_only_edge_index = torch.stack([torch.LongTensor(sourceNodesCPY), torch.LongTensor(targetNodesCPY)], dim=0)


else:
    data.train_pos_edge_index = torch.stack([torch.LongTensor(sourceNodes), torch.LongTensor(targetNodes)], dim=0)
    data.train_pos_only_edge_index = torch.stack([torch.LongTensor(sourceNodesCPY), torch.LongTensor(targetNodesCPY)], dim=0)

data.train_neg_edge_index = torch.stack([torch.LongTensor(negE[train_negIdx,0]), torch.LongTensor(negE[train_negIdx,1])], dim=0)
    #print(data.train_pos_edge_index.shape)
    
data.test_pos_edge_index = torch.stack([torch.LongTensor(posE[test_posIdx,0]), torch.LongTensor(posE[test_posIdx,1])], dim=0)

data.val_pos_edge_index = torch.stack([torch.LongTensor(posE[val_posIdx,0]), torch.LongTensor(posE[val_posIdx,1])], dim=0)


data.test_neg_edge_index = torch.stack([torch.LongTensor(negE[test_negIdx,0]), torch.LongTensor(negE[test_negIdx,1])], dim=0)

data.val_neg_edge_index = torch.stack([torch.LongTensor(negE[val_negIdx,0]), torch.LongTensor(negE[val_negIdx,1])], dim=0)


# In[5]:



channels = 128
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
h_sizes = [data.num_features]
for i in reversed(range(opts.hidden-1)):
    h_sizes.append((i+2)*channels)
    
h_sizes.append(channels)
#model = kwargs[modelName](Encoder(data.num_features, channels)).to(dev)
if opts.decoder == 'IP':
    model = GAEwithK(Encoder(h_sizes)).to(dev)
elif opts.decoder == 'TF':
    model = GAEwithK(Encoder(h_sizes), TFDecoder(data.num_nodes, TFNames.keys())).to(dev)
elif opts.decoder == 'RS':
    model = GAEwithK(Encoder(h_sizes), RESCALDecoder(channels)).to(dev)
else:
    print("ERR.")
    sys.exit()

x = data.x.to(dev)


train_pos_edge_index = data.train_pos_edge_index.to(dev)
train_pos_only_edge_index = data.train_pos_only_edge_index.to(dev)


train_neg_edge_index = data.train_neg_edge_index.to(dev)

test_pos_edge_index, test_neg_edge_index = data.test_pos_edge_index.to(dev), data.test_neg_edge_index.to(dev)

val_pos_edge_index, val_neg_edge_index = data.val_pos_edge_index.to(dev), data.val_neg_edge_index.to(dev)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
lossDict = {'epoch':[],'TrLoss':[], 'valLoss':  []}
last10Models = []
for epoch in tqdm(range(1, opts.epochs)):
    TrLoss = train()
    valLoss = val(val_pos_edge_index, val_neg_edge_index)
    #print(los.item())
    lossDict['epoch'].append(epoch)
    lossDict['TrLoss'].append(TrLoss.item())
    lossDict['valLoss'].append(valLoss.item())

    if np.mean(lossDict['valLoss'][-10:]) - valLoss.item() <= 1e-6 and epoch > 1000:
        break
z, epr, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
print(f"EPr: {epr}, AP: {ap}, Nodes: {len(NodeLst)}, Edges: {len(UnDirGr.edges())}",
      ", Test Edges:", len(np.unique(test_posIdx)),#data.test_pos_edge_index.shape[1],
      ", kTrain:",opts.kTrain,
      ", kTest:", opts.kTest,
      ", Type:", opts.encoder+'-'+opts.decoder+'-'+opts.type,
      ", ExpressionData:", opts.expFile,
      ", Network:", opts.netFile,
      ", CV_ID:", opts.id,
      ", rand_ID:",opts.rand,
      ", NumGCNs:", opts.hidden)
#pd.DataFrame.from_dict(lossDict).to_csv('lossDF.csv', index = False)
#print(model.decoder(z, [[2,10,810,12],[10,2,12,14]], sigmoid=True))

#print(model)
