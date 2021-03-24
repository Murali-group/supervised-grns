import os
import pandas as pd
from pathlib import Path
import numpy as np
import networkx as nx
import random
import math

from itertools import combinations, permutations, product
from tqdm import tqdm 
from sklearn.model_selection import KFold


import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv, GMMConv, GATConv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import numpy as np
import torchvision


from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected, negative_sampling

from SGRN.GAEHelpers import *






def run(RunnerObj, fID):
    '''
    Function to run GCN algorithm
    Requires the decoder parameter
    '''
    random.seed(RunnerObj.randSeed)
    np.random.seed(RunnerObj.randSeed)
    torch.manual_seed(RunnerObj.randSeed)

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
        epr, ap, auc_score, pred, act = model.test(z, pos_edge_index, neg_edge_index)
        return z, epr, ap, auc_score, pred, act

    def val(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        return loss

    print("Running fold: ", fID)
    
    print("Reading necessary input files...")
    
    exprDF = pd.read_csv(RunnerObj.inputDir.joinpath("temp/normExp.csv"), header = 0, index_col =0)
    posE = np.load(RunnerObj.inputDir.joinpath("temp/posE.npy"))
    negE = np.load(RunnerObj.inputDir.joinpath("temp/negE.npy"))
    nodeDict = np.load(RunnerObj.inputDir.joinpath("temp/nodeDict.npy"), allow_pickle = True)
    
    
    foldData = np.load(RunnerObj.inputDir.joinpath("temp/fold"+str(fID)+".npy"), allow_pickle = True)
    
    train_posIdx = foldData.item().get('train_posIdx')
    test_posIdx = foldData.item().get('test_posIdx')
    
    train_negIdx = foldData.item().get('train_negIdx')
    test_negIdx = foldData.item().get('test_negIdx')

    print("Done reading inputs...")
    
    val_posIdx = random.sample(list(train_posIdx), int(0.1*len(train_posIdx)))
    train_posIdx = list(set(train_posIdx).difference(set(val_posIdx)))

    val_negIdx = random.sample(list(train_negIdx), int(0.1*len(train_negIdx)))
    train_negIdx = list(set(train_negIdx).difference(set(val_negIdx)))
    
    sourceNodes = posE[train_posIdx , 0]
    targetNodes = posE[train_posIdx , 1]
    #print(sourceNodes)
    #print(targetNodes)
    
    subNodes = set(sourceNodes).union(set(targetNodes))
    allNodes = set(nodeDict.item().keys())
    missing = np.array(list(allNodes.difference(subNodes)))
    # find unlinked nodes and add self-edges
    sourceNodes = np.append(sourceNodes, missing)
    targetNodes = np.append(targetNodes, missing)
    #print(sourceNodes)
    #print(targetNodes)
    #print(missing)
    
    nodeFeatures = torch.Tensor(exprDF.values)

    if RunnerObj.params['encoder'] == 'GCN':
        eIndex = to_undirected(torch.LongTensor([sourceNodes, targetNodes]))
    elif RunnerObj.params['encoder'] == 'DGCN':
        eIndex = torch.LongTensor([sourceNodes, targetNodes])
    else:
        print("Invalid encoder name: ", RunnerObj.params.encoder)
        sys.exit()

    data = Data(x=nodeFeatures, edge_index=eIndex)

    if RunnerObj.params['encoder'] == 'GCN':
        data.train_pos_edge_index = to_undirected(torch.stack([torch.LongTensor(sourceNodes),
                                                               torch.LongTensor(targetNodes)], dim=0))
        data.train_pos_only_edge_index = torch.stack([torch.LongTensor(sourceNodes),
                                                      torch.LongTensor(targetNodes)], dim=0)

    elif RunnerObj.params['encoder'] == 'DGCN':
        data.train_pos_edge_index = torch.stack([torch.LongTensor(sourceNodes),
                                                 torch.LongTensor(targetNodes)], dim=0)
        data.train_pos_only_edge_index = torch.stack([torch.LongTensor(sourceNodes),
                                                      torch.LongTensor(targetNodes)], dim=0)
    else:
        print("Invalid encoder name: ", RunnerObj.params.encoder)
        sys.exit()

        

    data.test_pos_edge_index = torch.stack([torch.LongTensor(posE[test_posIdx,0]),
                                            torch.LongTensor(posE[test_posIdx,1])], dim=0)
    data.val_pos_edge_index = torch.stack([torch.LongTensor(posE[val_posIdx,0]),
                                           torch.LongTensor(posE[val_posIdx,1])], dim=0)
    
    data.train_neg_edge_index = torch.stack([torch.LongTensor(negE[train_negIdx,0]),
                                             torch.LongTensor(negE[train_negIdx,1])], dim=0)
    data.test_neg_edge_index = torch.stack([torch.LongTensor(negE[test_negIdx,0]),
                                            torch.LongTensor(negE[test_negIdx,1])], dim=0)
    data.val_neg_edge_index = torch.stack([torch.LongTensor(negE[val_negIdx,0]),
                                           torch.LongTensor(negE[val_negIdx,1])], dim=0)

    print("Done setting up data structures...")

    channels = RunnerObj.params['channels']
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h_sizes = [data.num_features]
    if RunnerObj.params['hidden'] >= 1:
        for i in reversed(range(RunnerObj.params['hidden']-1)):
            h_sizes.append((i+2)*channels)
        h_sizes.append(channels)

    #model = kwargs[modelName](Encoder(data.num_features, channels)).to(dev)
    if RunnerObj.params['decoder'] == 'IP':
        model = GAEwithK(Encoder(h_sizes)).to(dev)
    elif RunnerObj.params['decoder'] == 'NW':
        model = GAEwithK(Encoder(h_sizes), TFDecoder(data.num_nodes, TFNames.keys())).to(dev)
    elif RunnerObj.params['decoder'] == 'RS':
        model = GAEwithK(Encoder(h_sizes), RESCALDecoder(channels)).to(dev)
    else:
        print("Invalid decoder name:", RunnerObj.params['decoder'])
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
    print("Running  %s-%s..." %(RunnerObj.params['encoder'],RunnerObj.params['decoder']))

    for epoch in tqdm(range(1, RunnerObj.params['epochs'])):
        TrLoss = train()
        valLoss = val(val_pos_edge_index, val_neg_edge_index)
        #print(los.item())
        lossDict['epoch'].append(epoch)
        lossDict['TrLoss'].append(TrLoss.item())
        lossDict['valLoss'].append(valLoss.item())

        if np.mean(lossDict['valLoss'][-10:]) - valLoss.item()<= 1e-6 and epoch > 1000:
            break

    z, epr, ap, auc_score, preds, act = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print(epr, ap, auc_score)
    return act, preds
    
def parseOutput(RunnerObj):
    outDir = RunnerObj.outputDir
    
    if not Path(outDir).joinpath("outFile.txt").exists():
        print(outDir+'outFile.txt'+'does not exist, skipping...')
        return

