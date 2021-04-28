
import os
import pandas as pd
from pathlib import Path
import numpy as np
import networkx as nx
import random
import math
import random

from itertools import combinations, permutations, product
from tqdm import tqdm 
from sklearn.model_selection import KFold

def generateInputs(RunnerObj):
    '''
    If the folder/files under RunnerObj.datadir exist, 
    this function will not do anything.
    '''
    random.seed(RunnerObj.randSeed)
    np.random.seed(RunnerObj.randSeed)
    
    print(RunnerObj.__dict__)
    
    if not RunnerObj.inputDir.joinpath("{}CV".format(RunnerObj.CVType)).exists():
        print("Input folder does not exist: creating inputs")
        RunnerObj.inputDir.joinpath("{}CV".format(RunnerObj.CVType)).mkdir(exist_ok = False)
    else:
        print("Input folder exists... \n")

    
    # Check if all necesasary input files exist
    # Else recreate them
    
    fileList = ["GeneTFs.npy",
                "negE.npy",
                "negY.npy",
                "nodeDict.npy",
                "normExp.csv",
                "posE.npy",
                "posX_CNNC.npy",
                "posX.npy",
                "posY.npy"]
    fileFlag = True
    for fileName in fileList:
        if not os.path.isfile(RunnerObj.inputDir / fileName):
            #print(RunnerObj.inputDir / fileName)
            fileFlag = False
     
    if fileFlag:
        print("Reading input files...")
        exprDF = pd.read_csv(RunnerObj.inputDir.joinpath("normExp.csv"), header = 0, index_col =0)
        posE = np.load(RunnerObj.inputDir.joinpath("posE.npy"))
        negE = np.load(RunnerObj.inputDir.joinpath("negE.npy"))
        nodeDict = np.load(RunnerObj.inputDir.joinpath("nodeDict.npy"), allow_pickle = True)
        geneTFDict = np.load(RunnerObj.inputDir.joinpath("GeneTFs.npy"), allow_pickle = True)
        
        onlyTFs = geneTFDict.item().get('TF')
        onlyGeness = geneTFDict.item().get('Gene')
    else:
        print("Files not present, creating...")
        ExpDF = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                         header = 0, index_col = 0)
        GeneralChIP = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.trueEdges))
        # convert strings to upper case, just in case.
        GeneralChIP.Gene1 = GeneralChIP.Gene1.str.upper()
        GeneralChIP.Gene2 = GeneralChIP.Gene2.str.upper()

        # Remove self edges
        GeneralChIP = GeneralChIP[(GeneralChIP['Gene1']!= GeneralChIP['Gene2'])]
        # Only get subnetwork corresponding to genes in the expression data
        GeneralChIP = GeneralChIP[(GeneralChIP['Gene1'].isin(list(ExpDF.index))) & (GeneralChIP['Gene2'].isin(list(ExpDF.index)))]
        GeneralChIP = GeneralChIP.reset_index(drop = True)

        # Make an initial input graph
        DirGr = nx.from_pandas_edgelist(GeneralChIP,source='Gene1', target='Gene2', create_using = nx.DiGraph)
        newDirGr = nx.convert_node_labels_to_integers(DirGr,ordering = 'sorted', label_attribute = 'name')

        NodeLst = sorted(list(newDirGr.nodes()))

        onlyGenes = []
        onlyTFs = []
        TFNames = {}
        allNames = {}

        for n,data in newDirGr.nodes(data=True):
            allNames[n]=data['name']
            if data['name'] not in GeneralChIP.Gene1.values:
                onlyGenes.append(n)
            else:
                onlyTFs.append(n)
                TFNames[n] = data['name']

        # All possible edges
        possibleEdges = set(product(onlyTFs,NodeLst))

        # This order of nodes in newDirGr is same as DirGr
        # Double check
        NodeNames = sorted(list(DirGr.nodes()))
        subDF = ExpDF.loc[NodeNames]
        subDF = subDF.loc[~subDF.index.duplicated(keep='first')]
        subDFNorm = subDF.div(subDF.sum(axis=1), axis=0).fillna(0)

        # write normalized edpression 
        subDFNorm.to_csv(RunnerObj.inputDir.joinpath("normExp.csv"))

        pCnt = 0
        nCnt = 0

        # X, y for SVM, MLP
        posX = np.zeros((len(DirGr.edges()),subDF.shape[1]*2))    
        # X for CNNC
        posX_CNNC = np.zeros((len(DirGr.edges()), 32, 32))

        # y for CNNC, MLP, SVM
        posY = np.ones((len(DirGr.edges()), 1))
        negY = np.zeros((len(possibleEdges)-len(DirGr.edges()), 1))


        # Positive edge IDs
        posE = np.zeros((len(DirGr.edges()),2)).astype(int)
        # negative edge IDs
        negE = np.zeros((len(possibleEdges)-len(DirGr.edges()),2)).astype(int)
        #print(posX.shape, negX.shape, len(NodeLst))


        for edge in tqdm(possibleEdges):
            if edge in newDirGr.edges():
                posE[pCnt] = edge

                X = np.hstack([subDFNorm.iloc[edge[0]].values, subDFNorm.iloc[edge[1]].values])
                posX[pCnt] = X

                # !!!CNNC does not work woth normalized inputs, hence subDFNorm
                XC, xedges, yedges = np.histogram2d(subDF.iloc[edge[0]].values, subDF.iloc[edge[1]].values, bins = 32)       
                posX_CNNC[pCnt] = (np.log10(XC.T/subDF.shape[1] + 10 ** -4) + 4)/4

                pCnt += 1                
            else:
                negE[nCnt] = edge
                nCnt += 1

        np.save(RunnerObj.inputDir.joinpath("posE.npy"),posE.astype(int))
        np.save(RunnerObj.inputDir.joinpath("negE.npy"),negE.astype(int))

        # Feature vector for training in SVM, MLP; negX is too large
        np.save(RunnerObj.inputDir.joinpath("posX.npy"),posX)

        # Feature vector X  for training and testing in CNNC; negX is too large
        np.save(RunnerObj.inputDir.joinpath("posX_CNNC.npy"),posX_CNNC)

        np.save(RunnerObj.inputDir.joinpath("posY.npy"),posY.astype(int))
        np.save(RunnerObj.inputDir.joinpath("negY.npy"),negY.astype(int))


        np.save(RunnerObj.inputDir.joinpath("nodeDict.npy"),allNames)
        geneTFDict = {}
        geneTFDict['Gene']= onlyGenes
        geneTFDict['TF']= onlyTFs
        np.save(RunnerObj.inputDir.joinpath("GeneTFs.npy"),geneTFDict)

        print("Done writing input files needed for training and evaluation...")
        
    fileList = ["{}CV/fold-".format(RunnerObj.CVType)+str(RunnerObj.randSeed)+"-"+str(fID)+".npy" for fID in range(RunnerObj.kFold)]
    fileFlag = True
    for fileName in fileList:
        if not os.path.isfile(RunnerObj.inputDir / fileName):
            #print(RunnerObj.inputDir / fileName)
            fileFlag = False
    if fileFlag:
        print("Fold files exist. Skipping...")
        return 
    if RunnerObj.CVType == 'Edge':
        print("Creating folds for edge CV: ")

        # Create folds
        cv = KFold(n_splits=RunnerObj.kFold, random_state=RunnerObj.randSeed, shuffle=True)
        for fID in range(RunnerObj.kFold):
            iCnt = 0
            print("Writing inputs for fold:", fID)
            for train_index, test_index in cv.split(posE):
                if iCnt == fID:
                    train_posIdx = train_index
                    test_posIdx = test_index
                    break
                iCnt +=1
            iCnt = 0
            for train_index, test_index in cv.split(negE):
                if iCnt == fID:
                    train_negIdx = train_index
                    test_negIdx = test_index
                    break
                iCnt +=1

            test_negIdx = random.sample(list(test_negIdx), RunnerObj.kTest*len(test_posIdx))
            train_negIdx = random.sample(list(train_negIdx), RunnerObj.kTrain*len(train_posIdx))

            # Set of negatives used in training or testing
            #usedNeg = set(train_negIdx).union(set(test_negIdx))
            #freeNeg = set(range(len(negE))).difference(usedNeg)

            #print(len(negE), len(usedNeg), len(freeNeg))
            #print(len(train_posIdx), len(train_negIdx))

            # Important: Remove edges of type b->a from training set (positive and negative),
            # if a->b is in the test set (positive or negative)

            teEdgesInverse = set([])
            for idx in test_posIdx:
                teEdgesInverse.add((posE[idx , 1], posE[idx , 0]))

            for idx in test_negIdx:
                teEdgesInverse.add((negE[idx , 1], negE[idx , 0]))

            removeTrPos = set([])
            for idx in train_posIdx:
                if (posE[idx, 0], posE[idx, 1]) in teEdgesInverse:
                    # If the inverse edge is a training positive, add it to list
                    removeTrPos.add(idx)

            train_posIdx = list(set(train_posIdx).difference(removeTrPos))

            removeTrNeg = set([])
            for idx in train_negIdx:
                if (negE[idx, 0], negE[idx, 1]) in teEdgesInverse:
                    # If the inverse edge is a training positive, add it to list
                    removeTrNeg.add(idx)

            train_negIdx = list(set(train_negIdx).difference(removeTrNeg))

            foldDict = {}
            foldDict['train_posIdx'] = train_posIdx
            foldDict['train_negIdx'] = train_negIdx
            foldDict['test_posIdx'] = test_posIdx
            foldDict['test_negIdx'] = test_negIdx
            np.save(RunnerObj.inputDir.joinpath("{}CV/fold-".format(RunnerObj.CVType)+str(RunnerObj.randSeed)+"-"+str(fID)+".npy"),foldDict)
            
    elif RunnerObj.CVType == 'Node':
        
        # Create folds
        cv = KFold(n_splits=RunnerObj.kFold, random_state=RunnerObj.randSeed, shuffle=False)
        for fID in range(RunnerObj.kFold):
            iCnt = 0
            print("Writing inputs for fold:", fID)
            for train_index, test_index in cv.split(onlyTFs):
                if iCnt == fID:
                    #print(train_index, test_index, onlyTFs)
                    sNodes = [onlyTFs[x] for x in test_index]
                    break
                iCnt +=1

            test_posIdx = np.array([],dtype = int)
            #count positives for each edge
            tfIDDict = {}
            for tfID in sNodes:
                newArr = np.where(np.isin(posE,tfID))[0]
                test_posIdx = np.hstack((test_posIdx,newArr))
                tfIDDict[tfID] = len(newArr)
            test_posIdx = np.unique(test_posIdx)
            train_posIdx = np.setdiff1d(np.arange(0,len(posE)),test_posIdx)
            
            all_test_negIdx = np.array([],dtype = int)
            test_negIdx = np.array([],dtype = int)
            for tfID in sNodes:
                newArr = np.where(np.isin(negE,tfID))[0]
                random.shuffle(newArr)
                nTest = min(RunnerObj.kTest*tfIDDict[tfID],len(newArr))
                all_test_negIdx = np.hstack((all_test_negIdx,newArr))
                test_negIdx = np.hstack((test_negIdx,newArr[:nTest]))

            test_negIdx = np.unique(test_negIdx)
            all_train_negIdx =np.setdiff1d(np.arange(0,len(negE)), all_test_negIdx)
            nTrain = min(RunnerObj.kTrain*len(train_posIdx),len(all_train_negIdx))
            train_negIdx = np.random.choice(all_train_negIdx,nTrain) 
            
            foldDict = {}
            foldDict['train_posIdx'] = train_posIdx
            foldDict['train_negIdx'] = train_negIdx
            foldDict['test_posIdx'] = test_posIdx
            foldDict['test_negIdx'] = test_negIdx
            np.save(RunnerObj.inputDir.joinpath("{}CV/fold-".format(RunnerObj.CVType)+str(RunnerObj.randSeed)+"-"+str(fID)+".npy"),foldDict)
    else:
        print("CVType must either be 'Edge' or 'Node'")
        sys.exit()
    print("Done writing inputs")
    return