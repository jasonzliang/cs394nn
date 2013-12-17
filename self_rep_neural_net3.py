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
from sklearn.neighbors import NearestNeighbors

class noveltySearch:
  
  def __init__(self):
    self.archive = np.zeros((1,6), dtype=np.float)
    self.bestFitness = 1e308
    self.bestParams = None
    self.kdTree = NearestNeighbors(n_neighbors=10)
    self.p_min = 0.5
    self.numAdd = 0
    self.numNotAdd = 0
    self.scaleFactor = 1.0
    
  def computeNovelty(self, vector, params):
    if self.archive.shape[0] < 10:
      n = self.archive.shape[0]
    else:
      n = 10
      
    self.kdTree.fit(self.archive)
    dist, ind = self.kdTree.kneighbors(vector, n, return_distance=True)
    
    p = np.sum(dist)/n
    
    # add to archive if larger than p_min
    if p > self.p_min:
      self.archive = np.vstack((self.archive, vector))
      self.numAdd += 1
      self.numNotAdd = 0
    else:
      self.numAdd = 0
      self.numNotAdd +=1
    
    # adjust p_min
    if self.numAdd > 1:
      self.p_min *= 1.1
    if self.numNotAdd > 32 and self.p_min > 0.05:
      self.p_min *= 0.9
    
    fitness = vector[0]
    
    if self.bestFitness > fitness:
      self.bestFitness = fitness
      self.bestParams = params
    
#     print self.archive
    print self.numAdd, self.numNotAdd
    print self.bestFitness, self.archive.shape[0], self.p_min
      
    return p
  
class neuralWrapper:
  
  def __init__(self, percentage=0.9, structure=[8,12,12,1]):
    self.networkStructure = structure
    self.myNetwork = self.generateNetwork(self.networkStructure)
    self.myNetwork._setParameters(np.random.uniform(low=-1.0, high=1.0, size=len(self.myNetwork.params)))
    self.indices = self.freezedWeightsIndices(self.myNetwork, percentage)
    self.originalWeights = np.copy(self.myNetwork.params)
    self.noveltySearch = noveltySearch()
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
    sumoferror = self.sumOfErrors(weights)
    entrophy = self.getEntrophy(weights)
    median = np.median(weights)
    mean = np.mean(weights)
    stdev = np.std(weights)
    range = np.max(weights) - np.min(weights)
    
    vector = np.array([sumoferror, entrophy, median, mean, stdev, range])
    novelty = self.noveltySearch.computeNovelty(vector, self.myNetwork.params)
    
    return novelty
    
  def sumOfErrors(self, weights):
    self.myNetwork.params[self.indices] = weights
#     self.myNetwork._setParameters(self.originalWeights)
    error = 0.0
    for index, weight in enumerate(self.myNetwork.params):
      myInput = self.position2input(index)
      output = self.myNetwork.activate(myInput)[0]
      
      error += abs(weight - output)
    return error
  
  def getEntrophy(self, weights):
    term = 0.0
    for weight in weights:
      weight = abs(weight) + 0.00001
      term += math.log(weight)*weight
    return -1*term
  
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
    l = GA(self.fitnessFunction, self.myNetwork.params[self.indices])
    l.minimize = False
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
  for p in [1.0]:
    x = neuralWrapper(percentage=p)
    print "percentage:", p
    print "num connections:", x.myNetwork.params.shape[0]
    x.experiment1()
    x.compareWeights()

