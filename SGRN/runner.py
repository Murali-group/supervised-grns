import SGRN.GAERunner as GAE
import SGRN.makeInputs as mI


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
        
    def generateInputs(self):
        InputMapper[self.name](self)
        
        
    def run(self):
        for fID in range(self.params['kFolds']):
            AlgorithmMapper[self.name](self, fID)


    def parseOutput(self):
        OutputParser[self.name](self)