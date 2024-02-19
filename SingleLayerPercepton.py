import numpy as np

""""
@author: Gustavo Machado
"""

#Single layer percepton class

class percepton():
    def __init__(self, inputs, outputs):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__weights  = np.array([0.0, 0.0])

    def getInputs(self):
        return self.__inputs
    
    def getOutputs(self):
        return self.__outputs
    
    def getWeights(self):
        return self.__weights
    
    def setWeights(self, value, j):
        self.__weights[j] = value

    def StepFunction(self, SumFinal):
        if (SumFinal.any() >= 1):
            return 1
        return 0

    def OutputCalc(self, register):
        sum = register.dot(self.getWeights())
        return self.StepFunction(sum)

    def train(self, LearningRate):
        FinalMistake = 1
        while (FinalMistake != 0): 
            FinalMistake = 0
            for i in range(len(self.getOutputs())):
                CalculatedOutput = self.OutputCalc(np.asarray([self.getInputs()[i]]))
                mistake = abs(self.getOutputs()[i] - CalculatedOutput) 
                FinalMistake += mistake
                for j in range(len(self.getWeights())):
                    self.setWeights(((self.getWeights()[j]) + (LearningRate*self.getInputs()[i][j]*mistake)), j)
            print("Number of mistakes: " + str(FinalMistake))

#class trained for OR operation

OR = percepton(np.array([[0,0],[0,1], [1,0], [1,1]]), np.array([0,1,1,1]))
OR.train(0.1)
print("Neural Network trained")
print(OR.OutputCalc(OR.getInputs()[0]))
print(OR.OutputCalc(OR.getInputs()[1]))
print(OR.OutputCalc(OR.getInputs()[2]))
print(OR.OutputCalc(OR.getInputs()[3]))