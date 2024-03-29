import yaml
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import concurrent.futures
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

import torch
import random
import numpy as np
import SGRN as sr
import logging
import time
yaml.warnings({'YAMLLoadWarning': False})



def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(
        description='Run pathway reconstruction pipeline.')

    parser.add_argument('--config', default='config.yaml',
        help='Path to config file')

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

    try:
        opts = parse_arguments()
        config_file = opts.config


        with open(config_file, 'r') as conf:
            evaluation = sr.ConfigParser.parse(conf)
        print('\n Evaluation started\n')
        logging.info("Evaluation started")

        for idx in range(len(evaluation.runners)):
            evaluation.runners[idx].generateInputs()

        for idx in range(len(evaluation.runners)):
            start_time = time.process_time()
            logging.info("Training started for randSeed=%s with parameters=%s"%(evaluation.runners[idx].randSeed, str(evaluation.runners[idx].params)))
            evaluation.runners[idx].run()
            logging.info("Training completed in %.3f seconds"%(time.process_time()-start_time))
            
        for idx in range(len(evaluation.runners)):
            evaluation.runners[idx].parseOutput()

        logging.info("Evaluation complete!")
        print('\n Evaluation complete!\n')
    except:
        logging.exception('ERROR occurred...quitting. Check log file for errors.')
        raise


if __name__ == '__main__':
    main()
