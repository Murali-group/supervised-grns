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
    rSeed = RunnerObj.randSeed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(rSeed)
    np.random.seed(rSeed)
    torch.manual_seed(rSeed)
    torch.cuda.manual_seed(rSeed)
    
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
        yTrue, yPred = model.test(z, pos_edge_index, neg_edge_index)
        return yTrue, yPred#z, epr, ap, pred, act

    def val(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        return loss

    print("\n Running fold: ", fID)
    
    print("Reading necessary input files...")
    
    exprDF = pd.read_csv(RunnerObj.inputDir.joinpath("normExp.csv"), header = 0, index_col =0)
    posE = np.load(RunnerObj.inputDir.joinpath("posE.npy"))
    negE = np.load(RunnerObj.inputDir.joinpath("negE.npy"))
    nodeDict = np.load(RunnerObj.inputDir.joinpath("nodeDict.npy"), allow_pickle = True)
    geneTFDict = np.load(RunnerObj.inputDir.joinpath("GeneTFs.npy"), allow_pickle = True)
    onlyGenes = geneTFDict.item().get('Gene')
    onlyTFs = geneTFDict.item().get('TF')
                             
    foldData = np.load(RunnerObj.inputDir.joinpath("{}CV/fold-".format(RunnerObj.CVType)+str(RunnerObj.randSeed)+"-"+str(fID)+".npy"), allow_pickle = True)
    
    train_posIdx = foldData.item().get('train_posIdx')
    test_posIdx = foldData.item().get('test_posIdx')
    
    train_negIdx = foldData.item().get('train_negIdx')
    test_negIdx = foldData.item().get('test_negIdx')

    print("Done reading inputs...")
    
    val_posIdx = random.sample(list(train_posIdx), int(0.1*len(train_posIdx)))
    train_posIdx = list(set(train_posIdx).difference(set(val_posIdx)))

    val_negIdx = random.sample(list(train_negIdx), int(0.1*len(train_negIdx)))
    train_negIdx = list(set(train_negIdx).difference(set(val_negIdx)))
    #print(val_posIdx,val_negIdx)
    sourceNodes = posE[train_posIdx , 0]
    targetNodes = posE[train_posIdx , 1]
    
    presentNodesSet = set(sourceNodes).union(set(targetNodes))
    allNodes = set(nodeDict.item().keys())
    missingSet = allNodes.difference(presentNodesSet)
    presentNodes = np.array(list(presentNodesSet))
    missingNodes = np.array(list(missingSet))
    missingTFs = np.array(list(missingSet.intersection(set(onlyTFs))))
    presentTFs = np.array(list(set(sourceNodes)))

    #print(len(missing)*len(presentTF)+len(sourceNodes)+len(missingTF)*len(allNodes))

    # For missing TFs, additionally add edges outgoing to present nodes
    for tf in missingTFs:
        for node in presentNodes:
            sourceNodes = np.append(sourceNodes, tf)
            targetNodes = np.append(targetNodes, node)
    
    # find unlinked genes and TFs and have incoming edges from all TFs
    # Add edges from every TF to every gene that is missing from the network. 
    # This step helps to connect these genes to the network so that we can transfer information from TFs to these genes 
    # and potentially compute better embeddings for these genes.
    # Note: This is one of the ways to have them be part of the graph
    if RunnerObj.params['reconnect_disconnected_nodes']:
        for node in missingNodes:
            for tf in onlyTFs:
                sourceNodes = np.append(sourceNodes, tf)
                targetNodes = np.append(targetNodes, node)
    
            
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
        model = GAEwithK(Encoder(h_sizes), TFDecoder(data.num_nodes, onlyTFs)).to(dev)
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
        # Run until validation loss stabilizes after a certain minimum number of epochs.
        if np.mean(lossDict['valLoss'][-10:]) - valLoss.item()<= 1e-6 and epoch > 500:
            break
            
    yTrue, yPred = test(data.test_pos_edge_index, data.test_neg_edge_index)
    testIndices = torch.cat((data.test_pos_edge_index, data.test_neg_edge_index), axis=1).detach().cpu().numpy()
    edgeLength = testIndices.shape[1]
    outMatrix = np.vstack((testIndices, yTrue, yPred, np.array([fID]*edgeLength)))
    if not os.path.exists(RunnerObj.outPrefix):
        os.mkdir(RunnerObj.outPrefix)
    fullPath = Path(str(RunnerObj.outPrefix) + '/randID-' +  str(RunnerObj.randSeed) + '/' + RunnerObj.params['encoder'] + '-' +RunnerObj.params['decoder'])
    if not os.path.exists(fullPath):
        os.makedirs(fullPath)
    output_path =  fullPath / 'rankedEdges.csv'
    if os.path.isfile(output_path) and fID == 0:
        print("File exists, cleaning up")
        os.remove(output_path)
    outDF = pd.DataFrame(outMatrix.T, columns=['Gene1','Gene2','TrueScore','PredScore', 'CVID'])
    outDF = outDF.astype({'Gene1': int,'Gene2': int, 'CVID': int})
    #print(outDF.head())
    outDF['Gene1'] = outDF.Gene1.map(nodeDict.item())
    outDF['Gene2'] = outDF.Gene2.map(nodeDict.item())
    outDF.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
    
    

def parseOutput(RunnerObj):
    if RunnerObj.name == 'GAE':
        algName = RunnerObj.params['encoder'] + '-' +RunnerObj.params['decoder']
    else:
        algName = RunnerObj.name
           
    # Check if outfile exists
    fullPath = Path(str(RunnerObj.outPrefix) + '/randID-' +  str(RunnerObj.randSeed) + '/' + algName)
    if not os.path.isfile(fullPath/'rankedEdges.csv'):
        print("file does not exist, skipping:", fullPath/'rankedEdges.csv')
        return 
    
    inDF = pd.read_csv(fullPath/'rankedEdges.csv', index_col = None, header = 0)
    
    inDFAgg = inDF.sort_values('PredScore', ascending=False).drop_duplicates(subset=['Gene1','Gene2'], keep = 'first')
    
    # Write aggregated statistics
    inDFAgg.reset_index(inplace=True)    
    earlyPrecAgg = precision_at_k(inDFAgg.TrueScore, inDFAgg.PredScore, inDFAgg.TrueScore.sum())
    avgPrecAgg = average_precision_score(inDFAgg.TrueScore, inDFAgg.PredScore)
    statsAgg = Path(str(RunnerObj.outPrefix)) / "statsAggregated.csv".format(RunnerObj.randSeed)
    AUPRC, AUROC = computePRROC(inDFAgg.TrueScore, inDFAgg.PredScore)

    if os.path.isfile(statsAgg):
        outfile = open(statsAgg,'a')
        outfile.write('{},{},{},{},{},{},{},{},{}\n'.format(algName, RunnerObj.randSeed, earlyPrecAgg, avgPrecAgg,  AUPRC, AUROC,  inDFAgg.TrueScore.sum(), inDFAgg.shape[0],RunnerObj.CVType))
    else:
        outfile = open(statsAgg, 'w')
        outfile.write('Algorithm,randID,Early Precision,Average Precision,AUPRC,AUROC,#positives,#edges,CVType\n')
        outfile.write('{},{},{},{},{},{},{},{},{}\n'.format(algName, RunnerObj.randSeed, earlyPrecAgg, avgPrecAgg, AUPRC, AUROC, inDFAgg.TrueScore.sum(),  inDFAgg.shape[0],RunnerObj.CVType))
        
        
    # Write per-fold statistics
    for cvid in inDF.CVID.unique():
        subDF = inDF[inDF.CVID == cvid]
        earlyPrec = precision_at_k(subDF.TrueScore, subDF.PredScore, subDF.TrueScore.sum())
        a, b = computePRROC(subDF.TrueScore, subDF.PredScore)
        avgPrec = average_precision_score(subDF.TrueScore, subDF.PredScore)
        statsperFold = Path(str(RunnerObj.outPrefix)) / "statsperFold.csv".format(RunnerObj.randSeed)
    
        if os.path.isfile(statsperFold):
            outfile = open(statsperFold,'a')
            outfile.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(cvid, algName, RunnerObj.randSeed,
                                                       earlyPrec, avgPrec, AUPRC, AUROC, subDF.TrueScore.sum(),
                                                       subDF.shape[0], RunnerObj.CVType,RunnerObj.params))
        else:
            outfile = open(statsperFold, 'w')
            outfile.write('Fold,Algorithm,randID,Early Precision,Average Precision,AUPRC,AUROC,#positives,#edges,CVType\n')
            outfile.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(cvid, algName, RunnerObj.randSeed,
                                                       earlyPrec, avgPrec, AUPRC, AUROC,  subDF.TrueScore.sum(),
                                                       subDF.shape[0],RunnerObj.CVType,RunnerObj.params))
            
    return 

