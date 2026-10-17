import torch
import complexPyTorch
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_normalize, complex_relu


class TrainingFunctions:
    @staticmethod
    def trainingfunctioncaller(functionid,inputs):
        functionslist= [TrainingFunctions.sigmoidtest1,TrainingFunctions.sigmoidtest2,TrainingFunctions.sigmoidtest3]
        selectedfunction = functionslist[functionid]
        return selectedfunction(inputs)

    @staticmethod
    def sigmoidtest1(z):
        fn1 = torch.sin  # abs
        fn2 = torch.exp  # tan
        fn3 = torch.square
        sqr = torch.square
        # targetlinspace = self.dlfractal.generate2dcomplexlinspace(overridedimensions=(0,0)).flatten()
        target = torch.sigmoid(fn3(fn1(z * z + z / (z - 1)) * fn2(z * z - z / (z - 1)))) * 4
        return target

    @staticmethod
    def sigmoidtest2(z,verbose = False):
        fn1 = torch.sin  # abs
        fn2 = torch.square  # tan
        fn3 = torch.atan

        z = (fn1(z)+fn2(z))*fn3(z)
        z = z * z + z
        #batchnorm = ComplexBatchNorm1d(z.shape[0]).to(TrainingFunctions.getdevice())
        if verbose:
            print("sample of z before norm",z[25].item(),z[50].item(),z[100].item(),z[500].item(),z[1000].item())
            print("max z before",torch.max(z.real),torch.max(z.imag))
            print("min z before", torch.min(z.real), torch.min(z.imag))

        z = complex_normalize(z) * 4
        if verbose:
            print("sample of z after norm",z[25].item(),z[50].item(),z[100].item(),z[500].item(),z[1000].item())
            print("max z after", torch.max(z.real),torch.max(z.imag))
            print("min z after", torch.min(z.real), torch.min(z.imag))

        # targetlinspace = self.dlfractal.generate2dcomplexlinspace(overridedimensions=(0,0)).flatten()
        #target = torch.sigmoid(z)*4
        return z

    @staticmethod
    def sigmoidtest3(z, verbose=False):
        fn1 = torch.cos  # abs
        fn2 = torch.square  # tan
        fn3 = torch.cos

        z = (fn1(z) + fn2(z)) * fn3(z)
        z = z * z * z
        # batchnorm = ComplexBatchNorm1d(z.shape[0]).to(TrainingFunctions.getdevice())
        if verbose:
            print("sample of z before norm", z[25].item(), z[50].item(), z[100].item(), z[500].item(), z[1000].item())
            print("max z before", torch.max(z.real), torch.max(z.imag))
            print("min z before", torch.min(z.real), torch.min(z.imag))

        z = complex_normalize(z) * 4
        if verbose:
            print("sample of z after norm", z[25].item(), z[50].item(), z[100].item(), z[500].item(), z[1000].item())
            print("max z after", torch.max(z.real), torch.max(z.imag))
            print("min z after", torch.min(z.real), torch.min(z.imag))

        # targetlinspace = self.dlfractal.generate2dcomplexlinspace(overridedimensions=(0,0)).flatten()
        # target = torch.sigmoid(z)*4
        return z

    @staticmethod
    def getdevice():
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu:0')