#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import multiprocessing
from pathlib import Path


# local imports
import SGRNEval as ev 

def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(
        description='Run pathway reconstruction pipeline.')

    parser.add_argument('-c','--config', default='config.yaml',
        help="Configuration file containing list of datasets "
              "algorithms and output specifications.\n")
    
    parser.add_argument('-a', '--auc', action="store_true", default=False,
        help="Compute median of areas under Precision-Recall and ROC curves.\n")
    
        
    return parser

def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()
    
    return opts

def main():
    opts = parse_arguments()
    config_file = opts.config

    evalConfig = None

    with open(config_file, 'r') as conf:
        evalConfig = ev.ConfigParser.parse(conf)
        
    print('\nPost-run evaluation started...')
    evalSummarizer = ev.SGRNEval(evalConfig.input_settings, evalConfig.output_settings)
    
    outDir = os.path.join(str(evalSummarizer.output_settings.base_dir), \
            str(evalSummarizer.output_settings.output_prefix))

    # Compute and plot ROC, PRC and report median AUROC, AUPRC    
    if (opts.auc):
        print('\n\nComputing areas under ROC and PR curves...')

        AUPRC, AUROC = evalSummarizer.computeAUC()
        AUPRC.to_csv(os.path.join(outDir,'AUPRC.csv'))
        AUROC.to_csv(os.path.join(outDir,'AUROC.csv'))


    print('\n\nEvaluation complete...\n')


if __name__ == '__main__':
  main()