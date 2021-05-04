"""
SGRN Evaluation (:mod:`SGRNEval`) module contains the following
:class:`SGRNEval.SGRNEval` and three additional classes used in the
definition of SGRNEval class 
- :class:`SGRNEval.ConfigParser` 
- :class:`SGRNEval.InputSettings` 
- :class:`SGRNEval.OutputSettings`
"""
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
import concurrent.futures
from itertools import permutations
from collections import defaultdict



# local imports
from SGRNEval.computeAUC import PRROC


class InputSettings(object):
    '''
    The class for storing the names of input files.
    This initilizes an InputSettings object based on the
    following three parameters.
    
    :param datadir:   input dataset root directory, typically 'inputs/'
    :type datadir: str
    :param datasets:   List of dataset names
    :type datasets: list
        
    :param algorithms:   List of algorithm names
    :type algorithms: list
    '''

    def __init__(self,
            datadir, datasets, algorithms, randSeed) -> None:

        self.datadir = datadir
        self.datasets = datasets
        self.algorithms = algorithms
        self.randSeed = randSeed


class OutputSettings(object):
    '''    
    The class for storing the names of directories that output should
    be written to. This initilizes an OutputSettings object based on the
    following two parameters.
    
    :param base_dir: output root directory, typically 'outputs/'
    :type base_dir: str
    :param output_prefix: A prefix added to the final output files.
    :type str:
    '''

    def __init__(self, base_dir, output_prefix: Path) -> None:
        self.base_dir = base_dir
        self.output_prefix = output_prefix




class SGRNEval(object):
    '''
    The SGRN Evaluation object is created by parsing a user-provided configuration
    file. Its methods provide for further processing its inputs into
    a series of jobs to be run, as well as running these jobs.
    '''

    def __init__(self,
            input_settings: InputSettings,
            output_settings: OutputSettings) -> None:

        self.input_settings = input_settings
        self.output_settings = output_settings


    def computeAUC(self):

        '''
        Computes areas under the precision-recall (PR) and
        and ROC plots for each algorithm-dataset combination.
        
        :returns:
            - AUPRC: A dataframe containing AUPRC values for each algorithm-dataset combination
            - AUROC: A dataframe containing AUROC values for each algorithm-dataset combination
        '''
        AUPRCDict = {}
        AUROCDict = {}

        for dataset in tqdm(self.input_settings.datasets, 
                            total = len(self.input_settings.datasets), unit = " Datasets"):
            print("Evaluating for %s"%dataset)
            AUPRC, AUROC = PRROC(dataset, self.input_settings, 
                                    selfEdges = False)
            AUPRCDict[dataset['name']] = AUPRC
            AUROCDict[dataset['name']] = AUROC
        AUPRC = pd.DataFrame(AUPRCDict)
        AUROC = pd.DataFrame(AUROCDict)
        return AUPRC, AUROC
    

    

class ConfigParser(object):
    '''
    The class define static methods for parsing and storing the contents 
    of the config file that sets a that sets a large number of parameters 
    used in the SGRNEval.
    '''
    @staticmethod
    def parse(config_file_handle) -> SGRNEval:
        '''
        A method for parsing the input .yaml file.
        
        :param config_file_handle: Name of the .yaml file to be parsed
        :type config_file_handle: str
        
        :returns: 
            An object of class :class:`SGRNEval.SGRNEval`.
        '''
        config_map = yaml.load(config_file_handle)
        return SGRNEval(
            ConfigParser.__parse_input_settings(
                config_map['input_settings']),
            ConfigParser.__parse_output_settings(
                config_map['output_settings']))
    
    @staticmethod
    def __parse_input_settings(input_settings_map) -> InputSettings:
        '''
        A method for parsing and initializing 
        InputSettings object.
        '''
        input_dir = input_settings_map['input_dir']
        dataset_dir = input_settings_map['dataset_dir']
        datasets = input_settings_map['datasets']
        randSeed = input_settings_map['randSeed']

        return InputSettings(
                Path(input_dir, dataset_dir),
                datasets,
                ConfigParser.__parse_algorithms(
                input_settings_map['algorithms']),
                randSeed)


    @staticmethod
    def __parse_algorithms(algorithms_list):
        '''
        A method for parsing the list of algorithms
        that are being evaluated, along with
        any parameters being passed.
        
        Note that these parameters may not be
        used in the current evaluation, but can 
        be used at a later point.
        '''
        
        # Initilalize the list of algorithms
        algorithms = []
        
        # Parse contents of algorithms_list
        encoders = algorithms_list[0]['params']['encoder']
        decoders = algorithms_list[0]['params']['decoder']
        for encoder in encoders:
            for decoder in decoders:
                algorithms.append(encoder+"-"+decoder)

        return algorithms

    @staticmethod
    def __parse_output_settings(output_settings_map) -> OutputSettings:
        '''
        A method for parsing and initializing 
        Output object.
        '''
        output_dir = Path(output_settings_map['output_dir'])
        output_prefix = Path(output_settings_map['output_prefix'])

        return OutputSettings(output_dir,
                             output_prefix)