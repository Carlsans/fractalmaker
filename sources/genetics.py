import time
from datetime import timedelta

import pygad
import numpy as np
import torch
import sources.globalvariables as globalvariables
from sources import fractals
import torch.distributions.uniform




def getgeneticfitfrommodel(model,genenumber,numgeneration, populationcount, symetricentropy = -1):

    def fitness_function(solution, solution_idx):
        #print("ff sol=",solution[0],"index=",solution_idx)
        fixedparameters = torch.Tensor([-2, 2, -2, 2, 0, 0, 0])

        testedmodel = torch.concat((fixedparameters, torch.from_numpy(solution)), axis=0)
        if(model is not None):
            outputs = model(testedmodel)
            modelfitness = abs(10 - outputs.item())
        else:
            modelfitness = 0
        #entropy fitness
        FractalParameters = fractals.FractalParameters(xrange=(-2, 2), yrange=(-2, 2), max_iterations=40,
                                                             fractalbias=0.1 + 0.7j, fractalexponential=3,
                                                             fractaldimensions=(16, 16))
        fractal = fractals.Fractal(FractalParameters)
        fractal.parameters.setweighttensorfromparam(testedmodel)
        fitness = fractal.getfractalentropy(16,assymetric=symetricentropy) + modelfitness / 3
        #print(fitness)
        return fitness

    def mutation_func(offspring, ga_instance):
        if np.random.random() > .01:
            return offspring
        for chromosome_idx in range(offspring.shape[0]):
            random_gene_idx = np.random.choice(range(offspring.shape[1]))
            randommutator = (.5-np.random.random())/5
            if offspring[chromosome_idx, random_gene_idx] - abs(randommutator) > 0 and offspring[chromosome_idx, random_gene_idx] + randommutator < 1:
                offspring[chromosome_idx, random_gene_idx] += randommutator

        return offspring
    def callback_gen(ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])
        print("Population : ", ga_instance.population)





    #torch.manual_seed(1)

    initial_population = torch.rand(populationcount,genenumber,dtype=torch.double) # individuals,genes
    print("Initial population ",initial_population)
    ga_instance = pygad.GA(num_generations=numgeneration,
                           num_parents_mating=2,
                           parent_selection_type="sss",
                           random_mutation_min_val =0,
                           random_mutation_max_val=1,
                           mutation_type=mutation_func,
                            crossover_type="scattered",
                           allow_duplicate_genes=False,
                           fitness_func=fitness_function,
                            on_generation=callback_gen,
                           initial_population=initial_population)
    start = time.time()
    ga_instance.run()
    print("Final fitness of the best solution :", ga_instance.best_solution()[1])
    print("final vector: ", ga_instance.best_solution()[0])
    print(" Time elapsed for selecting individual : ", str(timedelta(seconds=time.time() - start)))
    return ga_instance.best_solution()[0]


def optimizefractalrange(fractal,fractaloriginalentropy,center,centertosidexyinitialrange,numgeneration=10, populationcount=200, entropyfuncprecision = 8,zoomfactor = 1,doubleprecision=False):
    genenumber = 2
    xcenter, ycenter = center
    centertosidexinitialrange, centertosideyinitialrange = centertosidexyinitialrange
    centertosidexzoomed,centertosideyzoomed = centertosidexinitialrange*zoomfactor, centertosideyinitialrange*zoomfactor
    meancenterlength = (centertosidexinitialrange + centertosideyinitialrange) / 2
    meanzoomedlength = (centertosidexzoomed+centertosideyzoomed) /2
    def fitness_function(solution, solution_idx):
        fractal.parameters.setrangefromcentertoside(center=(solution[0],solution[1]),centertosidexyinitialrange=(centertosidexzoomed,centertosideyzoomed))
        #fractal.parameters.xrange = (solution[0],solution[0] + (rangex*zoomfactor))
        #fractal.parameters.yrange = (solution[1], solution[1] + (rangey*zoomfactor))
        #fitness = abs(fractal.getfractalentropy(entropyfuncprecision) - fractaloriginalentropy)
        fitness =  fractal.getfractalentropy(entropyfuncprecision,doubleprecision,addcenter=True)
        #print(fitness)
        return fitness

    def mutation_func(offspring, ga_instance):
        if np.random.random() > .8:
            return offspring
        for chromosome_idx in range(offspring.shape[0]):
            random_gene_idx = np.random.choice(range(offspring.shape[1]))
            randommutator = (.5-np.random.default_rng().standard_normal(size=1, dtype='float32') )
            #if offspring[chromosome_idx, random_gene_idx] - abs(randommutator) > 0 and offspring[chromosome_idx, random_gene_idx] + randommutator < 1:
            offspring[chromosome_idx, random_gene_idx] += randommutator

        return offspring

    def callback_gen(ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])
        print("Population : ", ga_instance.population)
        if ga_instance.best_solution()[1] == 0:
            print("Fitness == 0, please restart with a new population")
            return "stop"





    #torch.manual_seed(1)
    #Solution is a point inside the initial range limit
    #xpopulation = torch.unsqueeze(torch.DoubleTensor(populationcount).uniform_(xcenter-centertosidexinitialrange, xcenter+centertosidexinitialrange),1)
    xpopulation = torch.arange(xcenter-centertosidexinitialrange,xcenter+centertosidexinitialrange,centertosidexinitialrange*2/populationcount,dtype=torch.double)
    xrand = torch.rand(populationcount,dtype=torch.double)*centertosidexinitialrange*2/populationcount
    xpopulation = torch.unsqueeze(torch.add(xpopulation,xrand),1)
    #ypopulation = torch.unsqueeze(torch.DoubleTensor(populationcount).uniform_(ycenter-centertosideyinitialrange, ycenter+centertosideyinitialrange),1)
    ypopulation = torch.arange(ycenter - centertosideyinitialrange, ycenter + centertosideyinitialrange,
                               centertosideyinitialrange * 2 / populationcount,dtype=torch.double)
    yrand = torch.rand(populationcount,dtype=torch.double) * centertosideyinitialrange * 2 / populationcount
    ypopulation = torch.unsqueeze(torch.add(ypopulation, yrand), 1)
    print("xpopulation",xpopulation)
    print("ypopulation", ypopulation)
    initial_population = torch.concat((xpopulation, ypopulation), axis=1)
    print("Initial population ",initial_population.shape)
    ga_instance = pygad.GA(num_generations=numgeneration,
                           num_parents_mating=4,
                           parent_selection_type="sss",
                           #random_mutation_min_val =0,
                           #random_mutation_max_val=1,
                           mutation_type=mutation_func,
                           #mutation_type=None,
                            crossover_type="scattered",
                           allow_duplicate_genes=False,
                           fitness_func=fitness_function,
                            on_generation=callback_gen,
                           initial_population=initial_population)
    start = time.time()
    ga_instance.run()
    print("Final fitness of the best solution :", ga_instance.best_solution()[1])
    print("final vector: ", ga_instance.best_solution()[0])
    print(" Time elapsed for selecting individual : ", str(timedelta(seconds=time.time() - start)))
    #fractal.parameters.xrange = (ga_instance.best_solution()[0][0], ga_instance.best_solution()[0][0] + (rangex*zoomfactor))
    #fractal.parameters.yrange = (ga_instance.best_solution()[0][1], ga_instance.best_solution()[0][1] + (rangey*zoomfactor))
    bestsolution = ga_instance.best_solution()[0]
    fractal.parameters.setrangefromcentertoside(center=(bestsolution[0], bestsolution[1]),
                                                centertosidexyinitialrange=(centertosidexzoomed, centertosideyzoomed))
    return (fractal.parameters,ga_instance.best_solution()[1])
