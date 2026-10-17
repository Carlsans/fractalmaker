import numpy as np
import torch
import sources.globalvariables as globalvariables


class FormulaInterpreter:
    formulatensor:torch.Tensor
    coefficienttensor:torch.Tensor
    formulastring:str
    formulavalue:complex
    formulamatrix:np.array
    coefficientmatrix:np.array
    c:complex
    z:complex
    def __init__(self,formulatensor,c=0):
        self.setnewformulatensor(formulatensor,0)
        self.operator = self.Operator(self)
        self.function = self.Function(self)
        self.variable = self.Variable(self)
    @staticmethod
    def isformulatensornormalized(tensor):
        for i in tensor:
            if i.item() < 0 or i.item()>1:
                raise Exception("Operator,Formula,Variable tensor value must be between 0 and 1.")

    def setnewformulatensor(self,formulatensor,c):
        self.isformulatensornormalized(formulatensor)
        # Reshape tensor to OFV(Operator,Formula,Variable) matrix
        try:
            reshapedtensor = torch.reshape(formulatensor, (-1, 7))
            #print(reshapedtensor)
            self.formulatensor = reshapedtensor[0:reshapedtensor.shape[0],0:3]
            self.coefficienttensor = reshapedtensor[0:reshapedtensor.shape[0],3:7]
            self.coefficientmatrix = self.coefficienttensor
        except RuntimeError:
            raise Exception("Operator,Formula,Variable vector must have a length divisible by 7")
        self.formulastring = ""
        self.formulavalue = 0
        self.c = c
        self.formulamatrix = np.zeros(self.formulatensor.shape)
        self.formulatensortoformulamatrix()
    def formulatensortoformulamatrix(self):
        # In form (V,F,O)
        NUM_OPERATORS = 5
        SINGLE_OPERATOR_PROBABILITY = 1 / NUM_OPERATORS
        NUM_FUNCTIONS = 20
        SINGLE_FUNCTION_PROBABILITY = 1 / NUM_FUNCTIONS
        NUM_VARIABLES = 22
        SINGLE_VARIABLE_PROBABILITY = 1 / NUM_VARIABLES
        #print(self.formulamatrix.shape)
        n = 0
        for i in self.formulatensor:
            #XXX

            variablenumber = self.floattoint(i[2], SINGLE_VARIABLE_PROBABILITY)
            functionnumber = self.floattoint(i[1], SINGLE_FUNCTION_PROBABILITY)
            operatornumber = self.floattoint(i[0], SINGLE_OPERATOR_PROBABILITY)
            self.formulamatrix[n] = [variablenumber,functionnumber,operatornumber]

            n+=1

    def tensortoOFV(self):
        #self.formulastring = ""
        #self.formulavalue = 0
        isfirst =True
        for i in self.formulatensor:

            # Get term operator, no change except for formulastring
            self.operator.operatorfromfloat(i[0],None,isfirst)
            # Get function, no change except for formulastring
            self.function.functionfromfloat(i[1],None)
            self.formulastring += "("
            # Get variable, update formulastring, store variable
            variable = self.variable.variablefromfloat(i[2])
            #print("variable",variable)
            self.formulastring += ") "
            # Apply function to the variable(no write to formulastring)
            lastvalue = self.function.functionfromfloat(i[1], variable)
            #print("afterfunction", lastvalue)
            # Apply the operator to the return of the function variable and save it to formulavalue
            self.operator.operatorfromfloat(i[0], lastvalue,isfirst,display=False)
            #print("formulavalueafteroperator",self.formulavalue)
            isfirst=False


    def setc(self,c):
        self.c = c
    def setz(self,z):
        self.z = z
    def computeformulavalue(self,c,z):
        self.formulastring = ""
        self.formulavalue=0
        self.setc(c)
        self.setz(z)
        self.tensortoOFV()
        #print(self.formulastring)
        #print(self.c)
        return self.formulavalue

    @staticmethod
    def floattoint(floatnumber,diviser):
        n = 0
        for i in np.arange (0, 1, diviser):
            if i > floatnumber:
                return n
            n += 1
        return n


    @staticmethod
    def getnormalizedvaluefromfloatinrange(floatnumber,lowerlimit,upperlimit,rangeincrement):
        lowerlimitfloat = lowerlimit * rangeincrement
        upperlimitfloat = upperlimit * rangeincrement
        range = upperlimitfloat - lowerlimitfloat
        normalizedvalue = (floatnumber - lowerlimitfloat) / range
        return normalizedvalue
    class Operator:
        NUM_OPERATORS = 4
        SINGLE_OPERATOR_PROBABILITY = 1 / NUM_OPERATORS
        def __init__(self,formulainterpreter):
            self.formulainterpreter = formulainterpreter

        def operatorfromfloat(self,floatnumber,lastvalue,isfirst,display=True):

            #print(floatnumber.dtype)
            operatornumber = self.formulainterpreter.floattoint(floatnumber, self.SINGLE_OPERATOR_PROBABILITY)
            if operatornumber == 1:
                if display:
                    self.formulainterpreter.formulastring += " + "
                if lastvalue != None:
                    self.formulainterpreter.formulavalue += lastvalue
            if operatornumber == 2:
                if display:
                    self.formulainterpreter.formulastring += " - "
                if lastvalue != None:
                    self.formulainterpreter.formulavalue -= lastvalue
            if operatornumber == 3:
                if isfirst:
                    if display:
                        self.formulainterpreter.formulastring += " + "
                    if lastvalue != None:
                        self.formulainterpreter.formulavalue += lastvalue
                else:
                    if display:
                        self.formulainterpreter.formulastring += " * "
                    if lastvalue != None:
                        self.formulainterpreter.formulavalue *= lastvalue


            if operatornumber == 4:
                if isfirst:
                    if display:
                        self.formulainterpreter.formulastring += " - "

                        if lastvalue != None:
                            self.formulainterpreter.formulavalue -= lastvalue
                else:
                    if display:
                        self.formulainterpreter.formulastring += " / "
                    if lastvalue != 0:
                        if lastvalue != None:
                            self.formulainterpreter.formulavalue /= lastvalue
                    else:
                        self.formulainterpreter.formulastring += "EDZ"
            return

    class Function:
        NUM_FUNCTIONS = 8
        SINGLE_FUNCTION_PROBABILITY = 1 / NUM_FUNCTIONS
        def __init__(self, formulainterpreter):
            self.formulainterpreter = formulainterpreter

        def functionfromfloat(self,floatnumber,variable):
            functionnumber = self.formulainterpreter.floattoint(floatnumber, self.SINGLE_FUNCTION_PROBABILITY)
            if functionnumber == 1:
                if variable is None:
                    self.formulainterpreter.formulastring += ""
                else:
                    return variable
            if functionnumber == 2:
                if variable is None:
                    self.formulainterpreter.formulastring += "sin"
                else:
                    return np.sin(variable)
            if functionnumber == 3:
                if variable is None:
                    self.formulainterpreter.formulastring += "cos"
                else:
                    return np.cos(variable)
            if functionnumber == 4:
                if variable is None:
                    self.formulainterpreter.formulastring += "tan"
                else:
                    return np.tan(variable)
            if functionnumber == 5:
                if variable is None:
                    self.formulainterpreter.formulastring += "asin"
                else:
                    return np.arcsin(variable)
            if functionnumber == 6:
                if variable is None:
                    self.formulainterpreter.formulastring += "acos"
                else:
                    return np.arccos(variable)
            if functionnumber == 7:
                if variable is None:
                    self.formulainterpreter.formulastring += "atan"
                else:
                    return np.arctan(variable)
            if functionnumber == 8:
                if variable is None:
                    self.formulainterpreter.formulastring += "zero"
                else:
                    return 0

    class Variable:
        NUM_VARIABLES = 5
        SINGLE_VARIABLE_PROBABILITY = 1 / NUM_VARIABLES
        def __init__(self, formulainterpreter):
            self.formulainterpreter = formulainterpreter

        def variablefromfloat(self,floatnumber):

            variablenumber = self.formulainterpreter.floattoint(floatnumber, self.SINGLE_VARIABLE_PROBABILITY)
            if variablenumber == 1:
                self.formulainterpreter.formulastring += "c"
                return self.formulainterpreter.c
            if variablenumber == 2:
                complexreal = self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber, variablenumber - 1,
                                                                           variablenumber,
                                                                           self.SINGLE_VARIABLE_PROBABILITY)
                compleximag = -self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber,
                                                                                         variablenumber - 1,
                                                                                         variablenumber,
                                                                                         self.SINGLE_VARIABLE_PROBABILITY)

                complexnumber = complex(complexreal, compleximag) + self.formulainterpreter.z
                self.formulainterpreter.formulastring += str(complexnumber) + "+z"
                return complexnumber
            if variablenumber == 3:
                complexreal = self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber, variablenumber - 1,
                                                                           variablenumber,
                                                                           self.SINGLE_VARIABLE_PROBABILITY)
                compleximag = -self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber,
                                                                                         variablenumber - 1,
                                                                                         variablenumber,
                                                                                         self.SINGLE_VARIABLE_PROBABILITY)

                complexnumber = complex(complexreal, compleximag) - self.formulainterpreter.z
                self.formulainterpreter.formulastring += str(complexnumber) + "-z"
                return complexnumber

            if variablenumber == 4:
                complexreal = self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber,
                                                                                         variablenumber - 1,
                                                                                         variablenumber,
                                                                                         self.SINGLE_VARIABLE_PROBABILITY)
                compleximag = -self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber,
                                                                                          variablenumber - 1,
                                                                                          variablenumber,
                                                                                          self.SINGLE_VARIABLE_PROBABILITY)
                complexnumber = complex(complexreal, compleximag) + self.formulainterpreter.c
                self.formulainterpreter.formulastring += str(complexnumber) + "+c"
                return complexnumber

            if variablenumber == 5:
                complexreal = self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber,
                                                                                         variablenumber - 1,
                                                                                         variablenumber,
                                                                                         self.SINGLE_VARIABLE_PROBABILITY)
                compleximag = -self.formulainterpreter.getnormalizedvaluefromfloatinrange(floatnumber,
                                                                                          variablenumber - 1,
                                                                                          variablenumber,
                                                                                          self.SINGLE_VARIABLE_PROBABILITY)
                complexnumber = complex(complexreal, compleximag) - self.formulainterpreter.c
                self.formulainterpreter.formulastring += str(complexnumber) + "-c"
                return complexnumber
        #def variablefromfloat(self,floatnumber):

# formulainterpreter = FormulaInterpreter(torch.rand(6),0)
#
# print(formulainterpreter.computecvalue(complex(.2,.2)))
# print(formulainterpreter.formulastring)
# print(formulainterpreter.computecvalue(complex(.1,.2)))
# print(formulainterpreter.formulastring)
# print(formulainterpreter.computecvalue(complex(.2,.1)))
# print(formulainterpreter.formulastring)
#
# #formulainterpreter.tensortoOFV()
# print("formulastring")

# print("formulavalue")
# print(formulainterpreter.formulavalue)