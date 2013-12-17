#!/usr/bin/python
'''
Created on Nov 16, 2013

@author: jason
'''
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.optimization import CMAES, NelderMead, ExactNES, FEM, StochasticHillClimber, GA
import csv, time, cPickle, math
import numpy as np
import random

class neuralWrapper:
  
  def __init__(self, percentage=0.9, structure=[8,12,12,1]):
    self.networkStructure = structure
    self.myNetwork = self.generateNetwork(self.networkStructure)
    self.myNetwork._setParameters(np.random.uniform(low=-1.0, high=1.0, size=len(self.myNetwork.params)))
    self.indices = self.freezedWeightsIndices(self.myNetwork, percentage)
    self.originalWeights = np.copy(self.myNetwork.params)
    self.metaInfo = {"percentage":percentage}
    assert 2**self.networkStructure[0] > len(self.myNetwork.params)
    
  def prettyPrintNet(self):
    net = self.myNetwork
    for mod in net.modules:
      print "Module:", mod.name
      if mod.paramdim > 0:
        print "--parameters:", mod.params
      for conn in net.connections[mod]:
        print "-connection to", conn.outmod.name
        if conn.paramdim > 0:
           print "- parameters", conn.params
      if hasattr(net, "recurrentConns"):
        print "Recurrent connections"
        for conn in net.recurrentConns:             
           print "-", conn.inmod.name, " to", conn.outmod.name
           if conn.paramdim > 0:
              print "- parameters", conn.params
  
  def generateNetwork(self, structure):
    n = FeedForwardNetwork()
    prevLayer = SigmoidLayer(structure[0])
    n.addInputModule(prevLayer)
    for index, num in enumerate(structure[1:-1]):
      tempLayer = SigmoidLayer(num)
      n.addModule(tempLayer)
      n.addConnection(FullConnection(prevLayer, tempLayer))
      prevLayer = tempLayer
     
    lastLayer = SigmoidLayer(structure[-1])
    n.addOutputModule(lastLayer)
    n.addConnection(FullConnection(prevLayer, lastLayer))
    n.sortModules()
    return n
  
  def freezedWeightsIndices(self, weights, percentage=0.9):
    numSamples = int(percentage*len(weights))
    return random.sample(xrange(len(weights)), numSamples)
  
  def fitnessFunction(self, weights):
    self.myNetwork.params[self.indices] = weights
#     self.myNetwork._setParameters(self.originalWeights)
    error = 0.0
    for index, weight in enumerate(self.myNetwork.params):
      myInput = self.position2input(index)
      output = self.myNetwork.activate(myInput)[0]
      
      error += abs(weight - output)
    return error
  
  def logNet(self, outname=None):
    if outname != None:
      f = open(outname, 'wb')
    else:
      f = open('results_' + str(self.networkStructure) + '_' + str(self.metaInfo["percentage"]) + '.pkl', 'wb')
    cPickle.dump((self.myNetwork, self.networkStructure, self.indices, self.originalWeights, self.metaInfo), f)
    f.close()
    
  def loadNet(self, filename):
    f = open(filename, 'rb')
    self.myNetwork, self.networkStructure, self.indices, self.originalWeights, self.metaInfo = cPickle.load(f) 
    print "num connections: ", self.myNetwork.params.shape[0]
    f.close()
  
  def position2input(self, pos):
    myInput = [int(x) for x in list('{0:0b}'.format(pos))]
    while len(myInput) < self.networkStructure[0]:
      myInput.insert(0,0)    
    return myInput
  
  def experiment1(self):
    l = CMAES(self.fitnessFunction, self.myNetwork.params[self.indices])
    l.minimize = True
    l.verbose = True
    l.maxLearningSteps = 500
    params, fitness = l.learn()
    self.myNetwork.params[self.indices] = params
    self.metaInfo["numsteps"] = l.maxLearningSteps
    self.metaInfo["fitness"] = fitness
#     self.myNetwork._setParameters(self.originalWeights)
    self.logNet()
  
  def compareWeights(self):
    error = 0.0
    for i,weight in enumerate(self.myNetwork.params):
      output = self.myNetwork.activate(self.position2input(i))[0]
      if i in self.indices:
        type = "unclamped"
      else:
        type = "clamped"
      print type + ", " + str(weight) + ", " + str(output)
      error += abs(weight - output)
    print "total error:", error
  

if __name__ == '__main__':
  for p in [1.0,0.9,0.8,0.6]:
    x = neuralWrapper(percentage=p)
    print "percentage:", p
    print "num connections:", x.myNetwork.params.shape[0]
    x.experiment1()
    x.compareWeights()

