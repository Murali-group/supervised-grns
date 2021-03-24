import SGRN.GAERunner as GAE
import SGRN.makeInputs as mI
from SGRN.GAEHelpers import compute_metrics


from pathlib import Path


InputMapper = {'GAE':mI.generateInputs,
                 }





AlgorithmMapper = {'GAE':GAE.run,
                  }




OutputParser = {'GAE':GAE.parseOutput, 
            }



class Runner(object):
    '''
    A runnable analysis to be incorporated into the pipeline
    '''
    def __init__(self,
                params):
        self.name = params['name']
        self.inputDir = params['inputDir']
        self.params = params['params']
        self.exprData = params['exprData']
        self.trueEdges = params['trueEdges']      
        self.kTrain = params['kTrain']      
        self.kTest = params['kTest']      
        self.randSeed = params['randSeed']
        self.actual = []
        self.predicted = []
        
    def generateInputs(self):
        InputMapper[self.name](self)
        
        
    def run(self):
        for fID in range(10):
            actual, predicted = AlgorithmMapper[self.name](self, fID)
            self.actual.append(actual)
            self.predicted.append(predicted)

        compute_metrics(self.actual, self.predicted)


    def parseOutput(self):
        OutputParser[self.name](self)