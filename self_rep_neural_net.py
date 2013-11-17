'''
Created on Nov 16, 2013

@author: jason
'''
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.optimization import CMAES, NelderMead, ExactNES, FEM
import csv, time, cPickle

def generateNetwork(structure):
#   n = buildNetwork(*structure, fast=False, outclass=SigmoidLayer)
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

networkStructure = [8,16,1]
myNetwork = generateNetwork(networkStructure)
assert 2**networkStructure[0] > len(myNetwork.params)

def fitnessFunction(weights):
#   print len(weights)
#   print len(myNetwork.params)
  myNetwork._setParameters(weights)

  error = 0.0
  
  for index, weight in enumerate(weights):
    myInput = [int(x) for x in list('{0:0b}'.format(index))]
    while len(myInput) < networkStructure[0]:
      myInput.insert(0,0)
    output = myNetwork.activate(myInput)[0]
    
    error += abs(weight - output)
  return error

def logNet():
  f = open('results_' + str(networkStructure) + '_' + str(time.time()) + '.pkl', 'wb')
  cPickle.dump(myNetwork, f)
  f.close()
  
def loadNet(filename):
  f = open(filename, 'rb')
  global myNetwork
  myNetwork = cPickle.load(f)
  f.close()

def position2input(pos):
  myInput = [int(x) for x in list('{0:0b}'.format(pos))]
  while len(myInput) < networkStructure[0]:
    myInput.insert(0,0)
    
  return myInput

def experiment1():
  l = CMAES(fitnessFunction, myNetwork.params)
  l.minimize = True
  l.verbose = True
  l.maxLearningSteps = 500
  params, fitness = l.learn()
  myNetwork._setParameters(params)
  logNet()

def experiment2():
  l = NelderMead(fitnessFunction, myNetwork.params)
  l.minimize = True
  l.verbose = True
  l.maxLearningSteps = 1000
  l.learn()
  logNet()
  
def experiment3():
  l = ExactNES(fitnessFunction, myNetwork.params)
  l.minimize = True
  l.verbose = True
  l.maxLearningSteps = 1000
  l.learn()
  logNet()
  
def experiment4():
  l = FEM(fitnessFunction, myNetwork.params)
  l.minimize = True
  l.verbose = True
  l.maxLearningSteps = 1000
  l.learn()
  logNet()
  
if __name__ == '__main__':
  experiment1()
#   loadNet('results_[8, 16, 1]_1384670399.89.pkl')
#   error = 0.0
#   for i,weight in enumerate(myNetwork.params):
#     output = myNetwork.activate(position2input(i))[0]
#     print weight, output
#     error += abs(weight - output)
#     
#   print error
#     n = generateNetwork([2,2,1])
#     n.params = [1,0,0,0,0,0]
#     print n.params