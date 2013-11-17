'''
Created on Nov 16, 2013

@author: jason
'''
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.optimization import CMAES

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

def experiment1():
  l = CMAES(fitnessFunction, myNetwork.params)
  l.minimize = True
  l.verbose = True
  l.learn()
  
if __name__ == '__main__':
  experiment1()
#     n = generateNetwork([2,2,1])
#     n.params = [1,0,0,0,0,0]
#     print n.params