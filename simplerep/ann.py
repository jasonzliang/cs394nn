import numpy
import math
import random

def assertSafe(a):
   if not isinstance(a, (float, int)):
      print("grep 9s8f2")
   if a == float("inf"):
      print("grep 09234nt")
   if a == float("-inf"):
      print("grep nvois4")
   if a == float("nan"):
      print("grep vxmok34")

EXP_ARG_MAX = 709.782
def sigmoid(x):
#   assertSafe(x)
   arg = -x
   y = 1 / (math.exp(arg) + 1) if arg < EXP_ARG_MAX else 0
#   assertSafe(y)
#   assert 0 <= y <= 1
   return y

def gaussian(x):
#   assertSafe(x)
   y = math.exp(-x * x)
#   assertSafe(y)
#   assert 0 <= y <= 1
   return y

def sinusoid(x):
#   assertSafe(x)
#   if abs(x) == float("inf"):
#      return 0
   s = math.sin(x)
   y = s * s
#   assertSafe(y)
#   assert 0 <= y <= 1
   return y

MAX_FLOAT_EXPONENT = 1023
MAX_MAGNITUDE = 1 << 20
def line(x):
   if x > MAX_MAGNITUDE:
      return MAX_MAGNITUDE
   if x < -MAX_MAGNITUDE:
      return -MAX_MAGNITUDE
   return x
   #return min(x, MAX_MAGNITUDE
   #return x if abs(x) < MAX_MAGNITUDE

activationFunctionPool = [line, sigmoid, gaussian, sinusoid]

#class MutatableArbitraryNeuralNetwork(ArbitraryNeuralNetwork):
#   def __init__(self):

class ArbitraryNeuralNetwork:
   def __init__(self, activationFunction):
      self.size = len(activationFunction)

      # each neuron has its own activation function
      self.activationFunction = list(activationFunction)

      self.activation = numpy.zeros(self.size)

      # self.weight[i] is the weights to neuron i
      shape = (self.size, self.size)
      self.weight = numpy.zeros(shape)

   def copy(self):
      net = ArbitraryNeuralNetwork(self.activationFunction)
      net.weight = self.weight.copy()
      return net

   def addNeuron(self):
      af = random.choice(activationFunctionPool)
      afAll = self.activationFunction
      self.activationFunction = numpy.append(afAll, [af], 0)

      self.activation = numpy.append(self.activation, [0], 0)

      z = numpy.zeros((1, self.size))
      zz = numpy.zeros((self.size + 1, 1))
      self.weight = numpy.concatenate((self.weight, z), 0)
      self.weight = numpy.concatenate((self.weight, zz), 1)

      self.size += 1

   def randomizeAllWeight(self, mean = 0, variance = 8):
      shape = (self.size, self.size)
      self.weight = numpy.random.normal(mean, variance, shape)

   def zeroAllActivation(self):
      self.activation = numpy.zeros(self.size)

   def setAllInput(self, inputs):
      n = min(len(inputs), self.size)
      for i in range(n):
         self.activation[i] = inputs[i]

   def getOutput(self, i):
      if 0 <= i < self.size:
         return self.activation[-1 - i]
      DEFAULT = 0
      return DEFAULT

   def step(self):
      # first we form the weighted sums
      self.activation = (self.weight).dot(self.activation)

      # allowing unbounded activation functions
      # allows infinty to appear.
      # hopefully this isn't a significant slow down
      #self.activation = numpy.nan_to_num(self.activation)

      # then we apply activation functions
      for i in range(len(self.activation)):
         f = self.activationFunction[i]
         self.activation[i] = f(self.activation[i])

      #inf = float("inf")
      #minf = -inf
      #nan = float("NaN")
      #assert all(a != inf for a in self.activation)
      #assert all(a != minf for a in self.activation)
      #assert all(a != nan for a in self.activation)
