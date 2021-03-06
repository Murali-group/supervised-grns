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
from sklearn.model_selection import train_test_split

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected, negative_sampling
import torch.utils.data as utils
from SGRN.GAEHelpers import *
import SGRN.GAERunner as GR




class CNNCNet(nn.Module):
    '''
    Model is derived from https://github.com/xiaoyeye/CNNC
    '''
    def __init__(self):
        super(CNNCNet, self).__init__()

        self.cnn1 = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
        
        self.dropout1 = nn.Dropout2d(p=0.5)    
        
        self.cnn2 = nn.Sequential(
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
        
        self.dropout2 = nn.Dropout2d(p=0.5)    

        self.cnn3 = nn.Sequential(
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(),
        nn.Flatten())        
        
        self.dropout3 = nn.Dropout2d(p=0.5)    
        

        self.fc1 = nn.Linear(128, 1)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.dropout1(x)

        x = self.cnn2(x)
        x = self.dropout2(x)

        x = self.cnn3(x)
        x = self.dropout3(x)
     
        x = F.sigmoid(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        return x


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)




def run(RunnerObj, fID):
    '''
    Function to run GCN algorithm
    Requires the decoder parameter
    '''


    print("\n Running fold: ", fID)
    
    print("Reading necessary input files...")
    
    #exprDF = pd.read_csv(RunnerObj.inputDir.joinpath("normExp.csv"), header = 0, index_col =0)
    # NOTE: CNNC requires non-standarized expression data as input 
    exprDF = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                         header = 0, index_col = 0)
    posE = np.load(RunnerObj.inputDir.joinpath("posE.npy"))
    negE = np.load(RunnerObj.inputDir.joinpath("negE.npy"))
    
    posY = np.load(RunnerObj.inputDir.joinpath("posY.npy"))
    negY = np.load(RunnerObj.inputDir.joinpath("negY.npy"))
    
    nodeDict = np.load(RunnerObj.inputDir.joinpath("nodeDict.npy"), allow_pickle = True)
    geneTFDict = np.load(RunnerObj.inputDir.joinpath("GeneTFs.npy"), allow_pickle = True)
    onlyGenes = geneTFDict.item().get('Gene')
    onlyTFs = geneTFDict.item().get('TF')
    posX = np.load(RunnerObj.inputDir.joinpath("posX_CNNC.npy"))
                         
    foldData = np.load(RunnerObj.inputDir.joinpath("{}CV/fold-".format(RunnerObj.CVType)+str(RunnerObj.randSeed)+"-"+str(fID)+".npy"), allow_pickle = True)
    
    train_posIdx = foldData.item().get('train_posIdx')
    test_posIdx = foldData.item().get('test_posIdx')
    
    train_negIdx = foldData.item().get('train_negIdx')
    test_negIdx = foldData.item().get('test_negIdx')

    print("Done reading inputs...")
    
    # Compute feature vectors for negatives 
    # storing the pre-computed negatives is prohibitive
    if RunnerObj.params['log']:
        print("Computing log transformed input features..")
        # If log is true recompute features for positives
        # This is specific to the dataset
        # CNNC datasets (m_Mac) need the log parameter to be True.
        
        # Step-1 features for training positives
        trPos =  np.zeros((len(train_posIdx), 32, 32))
        for idx in tqdm(range(len(train_posIdx))):
            edgeId = train_posIdx[idx]
            edge = posE[idx,:]
            nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
            X, xedges, yedges = np.histogram2d(np.log10(exprDF.loc[nEdge[0]].values+10**-2), np.log10(exprDF.loc[nEdge[1]].values+10**-2), bins = 32)
            trPos[idx,:] = (np.log10(X.T/exprDF.shape[1] + 10 ** -4) + 4)/4
            
        # Step-2 features for training negatives            
        trNeg =  np.zeros((len(train_negIdx), 32, 32))
        for idx in tqdm(range(len(train_negIdx))):
            edgeId = train_negIdx[idx]
            edge = negE[idx,:]
            nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
            X, xedges, yedges = np.histogram2d(np.log10(exprDF.loc[nEdge[0]].values+10**-2), np.log10(exprDF.loc[nEdge[1]].values+10**-2), bins = 32)
            trNeg[idx,:] = (np.log10(X.T/exprDF.shape[1] + 10 ** -4) + 4)/4

        # Step-3 features for test positives
        tePos =  np.zeros((len(test_posIdx), 32, 32))
        for idx in tqdm(range(len(test_posIdx))):
            edgeId = test_posIdx[idx]
            edge = posE[idx,:]
            nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
            X, xedges, yedges = np.histogram2d(np.log10(exprDF.loc[nEdge[0]].values+10**-2), np.log10(exprDF.loc[nEdge[1]].values+10**-2), bins = 32)
            tePos[idx,:] = (np.log10(X.T/exprDF.shape[1] + 10 ** -4) + 4)/4

            
        # Step-4 features for test negatives
        teNeg =  np.zeros((len(test_negIdx), 32, 32))
        for idx in tqdm(range(len(test_negIdx))):
            edgeId = test_negIdx[idx]
            edge = negE[idx,:]
            nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
            X, xedges, yedges = np.histogram2d(np.log10(exprDF.loc[nEdge[0]].values+10**-2), np.log10(exprDF.loc[nEdge[1]].values+10**-2), bins = 32)
            teNeg[idx,:] = (np.log10(X.T/exprDF.shape[1] + 10 ** -4) + 4)/4
            
        # Collect the datasets
        trainX = np.vstack((trPos,trNeg))
        testX = np.vstack((tePos,teNeg))
        
    else:
        print("Computing input features..")

        # If not "log", use posX for positives
        # Compute features for negative examples because
        # storing the pre-computed negatives is prohibitive

        trNeg =  np.zeros((len(train_negIdx), 32, 32))
        for idx in tqdm(range(len(train_negIdx))):
            edgeId = train_negIdx[idx]
            edge = negE[idx,:]
            nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
            X, xedges, yedges = np.histogram2d(exprDF.loc[nEdge[0]].values, exprDF.loc[nEdge[1]].values, bins = 32)
            trNeg[idx,:] = (np.log10(X.T/exprDF.shape[1] + 10 ** -4) + 4)/4



        teNeg =  np.zeros((len(test_negIdx), 32, 32))
        for idx in tqdm(range(len(test_negIdx))):
            edgeId = test_negIdx[idx]
            edge = negE[idx,:]
            nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
            X, xedges, yedges = np.histogram2d(exprDF.loc[nEdge[0]].values, exprDF.loc[nEdge[1]].values, bins = 32)
            teNeg[idx,:] = (np.log10(X.T/exprDF.shape[1] + 10 ** -4) + 4)/4

        trainX = np.vstack((posX[train_posIdx],trNeg))
        testX = np.vstack((posX[test_posIdx],teNeg))

        
    trainY = np.vstack((posY[train_posIdx],negY[train_negIdx]))
    testY = np.vstack((posY[test_posIdx],negY[test_negIdx]))
    
    # Generate validaiton dataset
    # 20% is the value used in CNNC paper.
    train_idx, val_idx = train_test_split(list(range(len(trainY))), test_size=0.2)
    tensorX = torch.from_numpy(trainX[train_idx]).float()
    tensorX = tensorX.unsqueeze(1)
    tensorY = torch.from_numpy(trainY[train_idx]).float()

    trainSet = utils.TensorDataset(tensorX, tensorY) # create your datset


    tensorX = torch.from_numpy(trainX[val_idx]).float()
    tensorX = tensorX.unsqueeze(1)
    tensorY = torch.from_numpy(trainY[val_idx]).float()

    valSet = utils.TensorDataset(tensorX, tensorY) # create your datset


    # https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/5
    batch_size = 1000

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size = batch_size, 
                                              shuffle = True, num_workers = 8)

    valloader = torch.utils.data.DataLoader(valSet, batch_size = len(valSet), 
                                              shuffle = True, num_workers = 8)


    tensorX1 = torch.from_numpy(testX).float()
    tensorX1 = tensorX1.unsqueeze(1)
    tensorY1 = torch.from_numpy(testY).float()

    testSet = utils.TensorDataset(tensorX1, tensorY1) # create your datset

    testloader = torch.utils.data.DataLoader(testSet, batch_size = len(testY), 
                                              shuffle = True, num_workers = 8)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = CNNCNet()
    net.apply(weight_init)
    net = net.to(device)


    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=1e-6, nesterov = True)
    
    # https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/5
    
    batch_size = 1000

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size = batch_size, 
                                              shuffle = True, num_workers = 8)

    valloader = torch.utils.data.DataLoader(valSet, batch_size = len(valSet), 
                                              shuffle = True, num_workers = 8)

    lossDict = {'epoch':[],'TrLoss':[], 'valLoss':  []}
    #print(test_posIdx, test_negIdx)
        
    for epoch in tqdm(range(RunnerObj.params['epochs'])):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(inputs)
                TrLoss = criterion(outputs, labels)
                TrLoss.backward()
                optimizer.step()

            running_loss += TrLoss.item()
            
        net.eval()
        for i, data1 in enumerate(valloader):
            inputs, labels = data1[0].to(device), data1[1].to(device)
            with torch.no_grad():
                outputs = net(inputs)
                valLoss = criterion(outputs, labels)
        
        #print(los.item())
        lossDict['epoch'].append(epoch)
        lossDict['TrLoss'].append(running_loss)
        lossDict['valLoss'].append(valLoss.item())
        # Run until validation loss stabilizes after a certain minimum number of epochs.
        if np.mean(lossDict['valLoss'][-10:]) - valLoss.item()<= 1e-6 and epoch > 500:
            break

     
    tensorX1 = torch.from_numpy(testX).float()
    tensorX1 = tensorX1.unsqueeze(1)
    tensorY1 = torch.from_numpy(testY).float()
    testIndices = torch.from_numpy(np.vstack((posE[test_posIdx,:], negE[test_negIdx,:]))).int()
    #print(testIndices)
    testSet = utils.TensorDataset(tensorX1, tensorY1, testIndices) # create your datset

    testloader = torch.utils.data.DataLoader(testSet, batch_size = len(testY), 
                                              shuffle = False, num_workers = 8)

    
    net.eval()

    
    for i, data2 in enumerate(testloader):
        inputs = data2[0].to(device)#, data2[1].to(device), data3[1].to(device)
        with torch.no_grad():
            outputs = net(inputs)
    
    yPred = outputs.detach().cpu().numpy()
    
    yTrue = data2[1].numpy()
    
    testIndices = data2[2].numpy()
    #print(testIndices)

    edgeLength = len(testY)
    #print(testIndices.shape, yTrue.shape, yPred.shape, np.reshape(np.array([fID]*edgeLength),(-1,1)))
    outMatrix = np.hstack((testIndices, yTrue, yPred, np.reshape(np.array([fID]*edgeLength),(-1,1))))
    if not os.path.exists(RunnerObj.outPrefix):
        os.mkdir(RunnerObj.outPrefix)
    fullPath = Path(str(RunnerObj.outPrefix) + '/randID-' +  str(RunnerObj.randSeed) + '/CNNC')
    if not os.path.exists(fullPath):
        os.makedirs(fullPath)
    output_path =  fullPath / 'rankedEdges.csv'
    if os.path.isfile(output_path) and fID == 0:
        print("File exists, cleaning up")
        os.remove(output_path)
    outDF = pd.DataFrame(outMatrix, columns=['Gene1','Gene2','TrueScore','PredScore', 'CVID'])
    outDF = outDF.astype({'Gene1': int,'Gene2': int, 'CVID': int})
    #print(outDF.head())
    outDF['Gene1'] = outDF.Gene1.map(nodeDict.item())
    outDF['Gene2'] = outDF.Gene2.map(nodeDict.item())
    outDF.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
    
    

def parseOutput(RunnerObj):
    GR.parseOutput(RunnerObj)
    return 

