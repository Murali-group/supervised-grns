"""
Supervised GRN Run (:mod:`SGRN`) module contains the following main class:
- :class:`SGRN.SGRN` and three additional classes used in the definition of SGRN class
- :class:`SGRN.ConfigParser`
- :class:`SGRN.InputSettings`
- :class:`SGRN.OutputSettings`
"""

import yaml
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import concurrent.futures
from typing import Dict, List
import yaml
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import concurrent.futures
from typing import Dict, List
from SGRN.runner import Runner
import os
import pandas as pd
import random
        
class InputSettings(object):
    def __init__(self,
            datadir, datasets, algorithms, kTrain, kTest, randSeeds, kFold, CVType) -> None:

        self.datadir = datadir
        self.datasets = datasets
        self.algorithms = algorithms
        self.kTrain = kTrain
        self.kTest = kTest
        self.randSeeds = randSeeds
        self.kFold = kFold
        self.CVType = CVType

class OutputSettings(object):
    '''
    Structure for storing the names of directories that output should
    be written to
    '''

    def __init__(self, base_dir, output_prefix: Path) -> None:
        self.base_dir = base_dir
        self.output_prefix = output_prefix

class SGRN(object):
    '''
    The SGRN object is created by parsing a user-provided configuration
    file. Its methods provide for further processing its inputs into
    a series of jobs to be run, as well as running these jobs.
    '''

    def __init__(self,
            input_settings: InputSettings,
            output_settings: OutputSettings) -> None:

        self.input_settings = input_settings
        self.output_settings = output_settings
        self.runners: Dict[int, Runner] = self.__create_runners()


    def __create_runners(self) -> Dict[int, List[Runner]]:
        '''
        Instantiate the set of runners based on parameters provided via the
        configuration file.
        '''
        
        runners: Dict[int, Runner] = defaultdict(list)
        order = 0
        for dataset in self.input_settings.datasets:
            for runner in self.input_settings.algorithms:
                for randSeed in self.input_settings.randSeeds:
                    data = {}
                    data['name'] = runner[0]
                    data['params'] = runner[1]
                    data['inputDir'] = Path.cwd().joinpath(self.input_settings.datadir.joinpath(dataset['name']))
                    data['exprData'] = dataset['exprData']
                    data['delim'] = dataset['delim']
                    data['trueEdges'] = dataset['trueEdges']
                    data['normalization'] = dataset['normalization']
                    data['min_genes'] = dataset['min_genes']
                    data['min_cells'] = dataset['min_cells']
                    data['top_expr_genes'] = dataset['top_expr_genes']
                    data['kTrain'] = self.input_settings.kTrain
                    data['kTest'] = self.input_settings.kTest
                    data['kFold'] = self.input_settings.kFold
                    data['randSeed'] = randSeed
                    data['outPrefix'] = self.output_settings.base_dir / self.output_settings.output_prefix
                    data['CVType'] = self.input_settings.CVType

                    if 'should_run' in data['params'] and \
                            data['params']['should_run'] is False:
                        print("Skipping %s" % (data['name']))
                        continue

                    runners[order] = Runner(data)
                    order += 1            
        return runners


    def execute_runners(self, parallel=False, num_threads=1):
        '''
        Run each of the algorithms.
        '''

        base_output_dir = self.output_settings.base_dir

        batches =  self.runners.keys()

        for batch in batches:
            if parallel==True:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                futures = [executor.submit(runner.run, base_output_dir)
                    for runner in self.runners[batch]]
                
                # https://stackoverflow.com/questions/35711160/detect-failed-tasks-in-concurrent-futures
                # Re-raise exception if produced
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                executor.shutdown(wait=True)
            else:
                for runner in self.runners[batch]:
                    runner.run(output_dir=base_output_dir)
                    
                    
class ConfigParser(object):
    '''
    Define static methods for parsing a config file that sets a large number
    of parameters for the pipeline
    '''
    @staticmethod
    def parse(config_file_handle) -> SGRN:
        config_map = yaml.load(config_file_handle)
        return SGRN(
            ConfigParser.__parse_input_settings(
                config_map['input_settings']),
            ConfigParser.__parse_output_settings(
                config_map['output_settings']))

    @staticmethod
    def __parse_input_settings(input_settings_map) -> InputSettings:
        input_dir = input_settings_map['input_dir']
        dataset_dir = input_settings_map['dataset_dir']
        datasets = input_settings_map['datasets']
        kTrain = input_settings_map['kTrain']
        kTest = input_settings_map['kTest']
        kFold = input_settings_map['kFold']
        CVType = input_settings_map['CVType']
        if 'randSeed' in input_settings_map:
            randSeeds = input_settings_map['randSeed']
        elif 'nTimes' in input_settings_map:
            randSeeds = []
            for x in range(input_settings_map['nTimes']):
                random.seed()
                randSeeds.append(random.randint(0, 1000))
            print("Using ",randSeeds, "as random seeds")
        else:
            print("Either randSeed or nTimes must be specified in the config file.")
            sys.exit()
            


        return InputSettings(
                Path(input_dir, dataset_dir),
                datasets,
                ConfigParser.__parse_algorithms(
                input_settings_map['algorithms']),
                kTrain, kTest, randSeeds, kFold, CVType)


    @staticmethod
    def __parse_algorithms(algorithms_list):
        algorithms = []
        for algorithm in algorithms_list:
                combos = [dict(zip(algorithm['params'], val))
                    for val in itertools.product(
                        *(algorithm['params'][param]
                            for param in algorithm['params']))]
                for combo in combos:
                    algorithms.append([algorithm['name'],combo])
            

        return algorithms

    @staticmethod
    def __parse_output_settings(output_settings_map) -> OutputSettings:
        output_dir = Path(output_settings_map['output_dir'])
        output_prefix = Path(output_settings_map['output_prefix'])

        return OutputSettings(output_dir,
                             output_prefix)


