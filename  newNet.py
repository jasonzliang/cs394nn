#!/usr/bin/python
'''
Created on Nov 16, 2013

@author: jason
'''
# from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.optimization import CMAES, NelderMead, ExactNES, FEM, StochasticHillClimber
import csv, time, cPickle, math
import numpy as np
import MultiNEAT as NEAT
from pprint import pprint

def dump(obj):
  for attr in dir(obj):
    print "obj.%s = %s" % (attr, getattr(obj, attr))

def evaluate(genome):

    # this creates a neural network (phenotype) from the genome

    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    
#     dump(net.InitRTRLMatrix.im_self)
    dump( net.InitRTRLMatrix.im_self.connections)
    # let's input just one pattern to the net, activate it once and get the output

    net.Input( [ 1.0, 0.0, 1.0 ] )
    net.Activate()
    output = net.Output() 

    # the output can be used as any other Python iterable. For the purposes of the tutorial, 
    # we will consider the fitness of the individual to be the neural network that outputs constantly 
    # 0.0 from the first output (the second output is ignored)

    fitness = 1.0 - output[0]
    return fitness
  
def main():
  params = NEAT.Parameters()  
  genome = NEAT.Genome(0, 3, 0, 2, False,
           NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)     
  pop = NEAT.Population(genome, params, True, 1.0)
  
  for generation in range(100): # run for 100 generations
    
    # retrieve a list of all genomes in the population
    genome_list = NEAT.GetGenomeList(pop)

    # apply the evaluation function to all genomes
    for genome in genome_list:
        fitness = evaluate(genome)
        genome.SetFitness(fitness)

    # at this point we may output some information regarding the progress of evolution, best fitness, etc.
    # it's also the place to put any code that tracks the progress and saves the best genome or the entire
    # population. We skip all of this in the tutorial. 

    # advance to the next generation
    pop.Epoch()

if __name__ == '__main__':
    main()