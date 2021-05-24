import os
import pandas as pd
from pathlib import Path
import numpy as np
import pdb
import scanpy as sc
import networkx as nx
import random
import math
import random

from itertools import combinations, permutations, product
from tqdm import tqdm 
from sklearn.model_selection import KFold

def exprdata_stats(adata):
    """ Printing Statistics of the single cell RNA sequence datasets including
    # of genes, # of cells
    """
    print(f'Total number of cells: {adata.n_obs}')
    print(f'Total number of genes: {adata.n_vars}')

def read_in_chunks(fileobj, chunksize=1000000):
    """
    Reading in a file lazily in increments of 1,000,000 bytes 1 Megabytes
    """
    while True:
        data = fileobj.read(chunksize)
        if not data:
            break
        yield data

def convert_transcripts_to_genes(RunnerObj):
    """ Converting transcripts to genes from Ensembl
    """
    info_df = []
    prev_line = ''
    prev_line2 = ''
    with open(os.path.join(RunnerObj.inputDir.joinpath(RunnerObj.gtf_file))) as f:
        for piece in read_in_chunks(f):
            piece = prev_line + piece
            for line in piece.split('\n'):
                if line == piece[piece.rfind('\n')+1:]:
                    prev_line = line
                    break
                if 'gene_name' in line and 'transcript_id' in line:
                    info = line[line.rfind('\t')+1:]
                    tmp = {}
                    for item in info.split('\"; '):
                        res = item.split(' \"')
                        if len(res) == 2:
                            tmp[res[0]] = res[1]
                    info_df.append([tmp['gene_name'],tmp['transcript_id']]) 
    tran_gene = pd.DataFrame(info_df)
    tran_gene.drop_duplicates(keep='first',inplace=True)
    tran_gene.columns = ['gene_name','transcript']
    tran_gene.set_index('transcript',inplace=True)
    return tran_gene

def transcription_factor_percentage(RunnerObj,expdf):
    """ Number of Transcription Factors in the single cell RNA seq dataset
    """
    tf_path = '/home/kradja/supervised-grns/inputs/TFs'
    tf_file = 'mouse-tfs.csv'
    tf = pd.read_csv(os.path.join(tf_path,tf_file),sep=',')

    tf_expdf = expdf[expdf.index.isin(tf.TF)]
    print(f'Out of {len(expdf)} genes there are {len(tf_expdf)} Transcription factors')
    pdb.set_trace()
    return tf_expdf
    

def preprocess_expr(RunnerObj):
    """
    Preprocessing single cell RNA sequencing data with inputs from the RunnerObj
    """
    ExpDF = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                 header = 0, index_col = 0,sep = RunnerObj.delim)
    adata = sc.read_csv(os.path.join(RunnerObj.inputDir.joinpath(RunnerObj.exprData))
                                     ,delimiter = RunnerObj.delim)
    adata = adata.transpose() # AnnData package needs the genes as columns and cells as rows
    exprdata_stats(adata)
    
    # Scanpy pre-processing filtering cells and genes and normlaization
    if RunnerObj.normalization == '':
        sc.pp.normalize_total(adata)
    sc.pp.filter_cells(adata,min_genes = int(RunnerObj.min_genes))
    sc.pp.filter_genes(adata,min_cells = int(RunnerObj.min_cells))
    exprdata_stats(adata)
    flavor = 'seurat'
    if flavor == 'seurat' or flavor == 'cell_ranger':
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,flavor = flavor)
    if flavor == 'seurat_v3':
        sc.pp.highly_variable_genes(adata,flavor = flavor)

    # Taking the top 'x' number of highly variable genes 
    high_var = adata.var[adata.var.highly_variable == True].sort_values('dispersions_norm',ascending=False)
    print(f'There are {len(high_var)} highly variable genes out of {len(adata.var)} genes')
    high_var = high_var[:int(RunnerObj.top_expr_genes)]
    adata_subset = adata[:, high_var.index]

    # Converting the indices of high_var with gene from transcript
    tran_gene = convert_transcripts_to_genes(RunnerObj)
    tt = adata_subset.var.index.to_series()
    ntt = tt.apply(lambda x: x[:x.rfind('.')])
    rr = ntt.map(tran_gene.to_dict()['gene_name']).fillna(ntt)
    adata_subset.var.index = rr

    # gene Irf3 shows up multiple times in the DataFrame. Multiple transcripts map to Irf3. How we take into account duplicates?
    expdf = pd.DataFrame(adata_subset.X,index = adata_subset.obs.index,columns = adata_subset.var.index)
    high_var_expdf = expdf.iloc[:,expdf.columns.isin(high_var.index)]

    return expdf.T, adata_subset.var

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
        ExpDF, expr_genes = preprocess_expr(RunnerObj)
        ExpDF.index = ExpDF.index.str.upper()

        # Percentage of TFs
        tf_expdf = transcription_factor_percentage(RunnerObj, ExpDF)

        # Replacing this line with a method that preprocesses exprData with scanpy
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

                # !!!CNNC does not work with normalized inputs, hence subDF
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
        cv = KFold(n_splits=RunnerObj.kFold, random_state=RunnerObj.randSeed, shuffle=True)
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
