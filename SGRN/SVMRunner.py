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
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
import SGRN.GAERunner as GR



def run(RunnerObj, fID):
    '''
    Function to run GCN algorithm
    Requires the decoder parameter
    '''


    print("\n Running fold: ", fID)
    
    print("Reading necessary input files...")
    
    exprDF = pd.read_csv(RunnerObj.inputDir.joinpath("normExp.csv"), header = 0, index_col =0)
    posE = np.load(RunnerObj.inputDir.joinpath("posE.npy"))
    negE = np.load(RunnerObj.inputDir.joinpath("negE.npy"))
    
    posY = np.load(RunnerObj.inputDir.joinpath("posY.npy"))
    negY = np.load(RunnerObj.inputDir.joinpath("negY.npy"))
    
    nodeDict = np.load(RunnerObj.inputDir.joinpath("nodeDict.npy"), allow_pickle = True)
    geneTFDict = np.load(RunnerObj.inputDir.joinpath("GeneTFs.npy"), allow_pickle = True)
    onlyGenes = geneTFDict.item().get('Gene')
    onlyTFs = geneTFDict.item().get('TF')
    posX = np.load(RunnerObj.inputDir.joinpath("posX.npy"))
                         
    foldData = np.load(RunnerObj.inputDir.joinpath("{}CV/fold-".format(RunnerObj.CVType)+str(RunnerObj.randSeed)+"-"+str(fID)+".npy"), allow_pickle = True)
    
    train_posIdx = foldData.item().get('train_posIdx')
    test_posIdx = foldData.item().get('test_posIdx')
    
    train_negIdx = foldData.item().get('train_negIdx')
    test_negIdx = foldData.item().get('test_negIdx')

    print("Done reading inputs...")
    
    # Compute feature vectors for negatives 
    # storing the pre-computed negatives is prohibitive
    print("Computing input features..")

    # If not "log", use posX for positives
    # Compute features for negative examples because
    # storing the pre-computed negatives is prohibitive

    trNeg =  np.zeros((len(train_negIdx), exprDF.shape[1]*2))
    for idx in tqdm(range(len(train_negIdx))):
        edgeId = train_negIdx[idx]
        edge = negE[idx,:]
        nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
        X = np.hstack((exprDF.loc[nEdge[0]].values, exprDF.loc[nEdge[1]].values))
        trNeg[idx,:] = X



    teNeg =  np.zeros((len(test_negIdx), exprDF.shape[1]*2))
    for idx in tqdm(range(len(test_negIdx))):
        edgeId = test_negIdx[idx]
        edge = negE[idx,:]
        nEdge = (nodeDict.item().get(edge[0]),nodeDict.item().get(edge[1]))
        X = np.hstack((exprDF.loc[nEdge[0]].values, exprDF.loc[nEdge[1]].values))
        teNeg[idx,:] = X

    trainX = np.vstack((posX[train_posIdx],trNeg))
    testX = np.vstack((posX[test_posIdx],teNeg))
        
    trainY = np.vstack((posY[train_posIdx],negY[train_negIdx]))
    testY = np.vstack((posY[test_posIdx],negY[test_negIdx]))
    
    svm = LinearSVC(max_iter=RunnerObj.params['maxIter'])
    clf = CalibratedClassifierCV(svm) 

    shuffleIdx = np.random.permutation(len(trainY))
    clf.fit(trainX[shuffleIdx,:], np.squeeze(trainY[shuffleIdx],1))
    outputs = clf.predict_proba(testX)[:,1]
     
    yPred = np.reshape(outputs,(-1,1))
    
    yTrue = testY
    
    testIndices = np.vstack((posE[test_posIdx,:], negE[test_negIdx,:]))

    #print(testIndices)

    edgeLength = len(testY)
    #print(testIndices.shape, yTrue.shape, yPred.shape, np.reshape(np.array([fID]*edgeLength),(-1,1)))
    outMatrix = np.hstack((testIndices, yTrue, yPred, np.reshape(np.array([fID]*edgeLength),(-1,1))))
    if not os.path.exists(RunnerObj.outPrefix):
        os.mkdir(RunnerObj.outPrefix)
    fullPath = Path(str(RunnerObj.outPrefix) + '/randID-' +  str(RunnerObj.randSeed) + '/SVM')
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

