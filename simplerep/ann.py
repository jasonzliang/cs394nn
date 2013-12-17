import numpy
import math
import random

EXP_ARG_MAX = 709.782
def sigmoid(x):
   arg = -x
   return 1 / (math.exp(arg) + 1) if arg < EXP_ARG_MAX else 0

def gaussian(x):
   return math.exp(-x * x)

def sinusoid(x):
   s = math.sin(x)
   return s * s

#MAX_FLOAT_EXPONENT = 1023
MAX_MAGNITUDE = float(1 << 20)
def line(x):
   if x > MAX_MAGNITUDE:
      return MAX_MAGNITUDE
   if x < -MAX_MAGNITUDE:
      return -MAX_MAGNITUDE
   return x

activationFunctionPool = [line, sigmoid, gaussian, sinusoid]

class ArbitraryNeuralNetwork:
   def __init__(self, activationFunction):
      self.size = len(activationFunction)

      # each neuron has its own activation function
      self.activationFunction = list(activationFunction)

      self.activation = numpy.zeros(self.size)

      # self.weight[i] is the weights to neuron i
      shape = (self.size, self.size)
      self.weight = numpy.zeros(shape)

      # XXX hacky addition
      self.frozen = False
      self.fitness = 0
      self.deepestDecode = self

   # XXX hacky addition
   def mutated(self):
      net = self.copy()

   def copy(self):
      net = ArbitraryNeuralNetwork(self.activationFunction)
      net.weight = self.weight.copy()
      return net

   def randomizeAllWeight(self, mean = 0, variance = 8):
      shape = (self.size, self.size)
      self.weight = numpy.random.normal(mean, variance, shape)

   def zeroAllActivation(self):
      self.activation = numpy.zeros(self.size)

   def setAllInput(self, inputs):
      # the first input is the first neuron
      n = min(len(inputs), self.size)
      for i in range(n):
         self.activation[i] = inputs[i]

   def getOutput(self, i):
      # the first output is the last neuron
      if 0 <= i < self.size:
         return self.activation[-1 - i]
      DEFAULT = 0
      return DEFAULT

   def step(self):
      # form the weighted sums
      self.activation = (self.weight).dot(self.activation)

      # infinity may appear due to large weights
      # or large or unbounded activation functions.
      # hopefully this isn't a significant slow down
      self.activation = numpy.nan_to_num(self.activation)

      # apply activation functions
      for i in range(len(self.activation)):
         f = self.activationFunction[i]
         self.activation[i] = f(self.activation[i])

