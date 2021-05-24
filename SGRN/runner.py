import SGRN.GAERunner as GAE
import SGRN.makeInputs as mI
import SGRN.CNNCRunner as CNNC
import SGRN.MLPRunner as MLP
import SGRN.SVMRunner as SVM


from pathlib import Path


InputMapper = {'GAE':mI.generateInputs,
               'CNNC': mI.generateInputs,
               'MLP': mI.generateInputs,
               'SVM': mI.generateInputs,

                 }





AlgorithmMapper = {'GAE':GAE.run,
                   'CNNC': CNNC.run,
                   'MLP': MLP.run,
                   'SVM': SVM.run,

                  }




OutputParser = {'GAE':GAE.parseOutput, 
                'CNNC':CNNC.parseOutput,
                'MLP':MLP.parseOutput,
                'SVM':SVM.parseOutput,
                
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
        self.delim = params['delim']
        self.normalization = params['normalization']
        self.min_genes = params['min_genes']
        self.min_cells = params['min_cells']
        self.top_expr_genes = params['top_expr_genes']
        self.gtf_file= params['gtf_file']
        self.trueEdges = params['trueEdges']      
        self.kTrain = params['kTrain']      
        self.kTest = params['kTest']      
        self.randSeed = params['randSeed']      
        self.outPrefix = params['outPrefix']
        self.kFold = params['kFold']
        self.CVType = params['CVType']
        
    def generateInputs(self):
        InputMapper[self.name](self)
        
        
    def run(self):
        for fID in range(self.kFold):
            AlgorithmMapper[self.name](self, fID)


    def parseOutput(self):
        OutputParser[self.name](self)
