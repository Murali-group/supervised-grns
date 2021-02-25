import SGRN.GAERunner as GAE


from pathlib import Path


InputMapper = {'GAE':GAE.generateInputs,
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
        self.cellData = params['cellData']
        
    def generateInputs(self):
        InputMapper[self.name](self)
        
        
    def run(self):
        AlgorithmMapper[self.name](self)


    def parseOutput(self):
        OutputParser[self.name](self)