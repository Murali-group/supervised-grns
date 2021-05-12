import pandas as pd
import numpy as np
#import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector


def PRROC(dataDict, inputSettings, selfEdges = False):
    '''
    Computes areas under the precision-recall and ROC curves
    for a given dataset for each algorithm.
    
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: boolPRROC
        
    :returns:
            - AUPRC: A dictionary containing AUPRC values for each algorithm
            - AUROC: A dictionary containing AUROC values for each algorithm
    '''
    
    # Initialize data dictionaries
    precisionDict = {}
    recallDict = {}
    FPRDict = {}
    TPRDict = {}
    AUPRC = {}
    AUROC = {}
    
    # set-up outDir that stores output directory name
    outDir = "outputs/"+dataDict['name']
    for algo in tqdm(inputSettings.algorithms, 
                         total = len(inputSettings.algorithms), unit = "Algorithms"):
        for rSeed in tqdm(inputSettings.randSeed, 
                         total = len(inputSettings.randSeed), unit = "Rand Seed"):
            # check if the output rankedEdges file exists
                if Path(outDir + '/randID-' + str(rSeed) + '/' + algo +'/rankedEdges.csv').exists():
                    rankedEdgesDF = pd.read_csv(outDir + '/randID-' + str(rSeed) + '/' + algo +'/rankedEdges.csv', \
                                                sep = ',', header =  0, index_col = None)
                    trueEdgesDF = rankedEdgesDF['TrueScore']
                    predDF = rankedEdgesDF['PredScore']


                    precisionDict[algo], recallDict[algo], FPRDict[algo], TPRDict[algo], AUPRC[algo], AUROC[algo] = computeScores(trueEdgesDF, predDF)   

                else:
                    print(outDir + '/randID-' + str(rSeed) + '/' + algo +'/rankedEdges.csv', \
                          ' does not exist. Skipping...')

    return AUPRC, AUROC

def computeScores(trueEdgesDF, predEdgeDF):
    '''        
    Computes precision-recall and ROC curves
    using scikit-learn for a given set of predictions in the 
    form of a DataFrame.
    
    :param trueEdgesDF:   A pandas dataframe containing the true classes.The indices of this dataframe are all possible edgesin a graph formed using the genes in the given dataset. This dataframe only has one column to indicate the classlabel of an edge. If an edge is present in the reference network, it gets a class label of 1, else 0.
    :type trueEdgesDF: DataFrame
        
    :param predEdgeDF:   A pandas dataframe containing the edge ranks from the prediced network. The indices of this dataframe are all possible edges.This dataframe only has one column to indicate the edge weightsin the predicted network. Higher the weight, higher the edge confidence.
    :type predEdgeDF: DataFrame
    
        
    :returns:
            - prec: A list of precision values (for PR plot)
            - recall: A list of precision values (for PR plot)
            - fpr: A list of false positive rates (for ROC plot)
            - tpr: A list of true positive rates (for ROC plot)
            - AUPRC: Area under the precision-recall curve
            - AUROC: Area under the ROC curve
    '''


    prroc = importr('PRROC')
    prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(trueEdgesDF.values)), 
              weights_class0 = FloatVector(list(predEdgeDF.values)), curve=True)


    fpr, tpr, thresholds = roc_curve(y_true=trueEdgesDF,
                                     y_score=predEdgeDF, pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=trueEdgesDF,
                                                      probas_pred=predEdgeDF, pos_label=1)
    
    return prec, recall, fpr, tpr, prCurve[1][0], auc(fpr, tpr)