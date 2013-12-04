import ann
import random
import math

# create a random net
def randomNet(nNeurons):
   pool = ann.activationFunctionPool
   af = [random.choice(pool) for i in range(nNeurons)]
   net = ann.ArbitraryNeuralNetwork(af)
   net.randomizeAllWeight()
   return net

def toTag(x):
   return math.floor(x)

def toActivationFunction(x):
   fPart, iPart = math.modf(x)

   pool = ann.activationFunctionPool
   i = int(fPart * len(pool))
   return pool[i]
   #int(x) * len(pool)
   #x = int(len(pool) * x) % len(pool)
   #return pool[x]

def decodeANN(net, maxSteps):
   net.zeroAllActivation()
   net.setAllInput([])

   tag = set()
   weight = dict()
   activationFunction = dict()

   # get all connections, neuron tags,
   # and neuron activation functions

   nSteps = 0
   while nSteps < maxSteps:
   #for i in range(MAX_STEPS):
      net.step()
      nSteps += 1
      #print(
      #   " ".join(".{:<2d}".format(int(10 * a))
      #      if abs(a) < 1
      #      else "e{:<2d}".format(int(math.copysign(math.log(abs(a), 10), a)))
      #   for a in net.activation))

      started = net.getOutput(0) > .00001
      finished = net.getOutput(1) > .99999

      #print(net.getOutput(0), net.getOutput(1), started, finished)

      if finished:
         break
      if not started:
         continue

      tagTarget = toTag(net.getOutput(2))
      activatorTarget = toActivationFunction(net.getOutput(3))

      tagSource = toTag(net.getOutput(4))
      activatorSource = toActivationFunction(net.getOutput(5))

      tag.add(tagTarget)
      tag.add(tagSource)

      weight[(tagTarget,tagSource)] = net.getOutput(6)

      activationFunction[tagTarget] = activatorTarget
      activationFunction[tagSource] = activatorSource

   #print("{:3d} steps".format(nSteps), end="  ")
   #print(len(tag), "tags")

   # inputs go from least positive to greatest positive.
   # outputs go from greatest negative to least negative.
   # arbitrarily many neurons sit in between
   tagNonNegative = sorted(t for t in tag if t >= 0)
   tagNegative = sorted(t for t in tag if t < 0)
   tag = tagNonNegative + list(reversed(tagNegative))

   # create a net
   #afNonNegative = [activationFunction[t] for t in tagNonNegative]
   #afNegative = [activationFunction[t] for t in tagNegative]
   #af = afNonNegative + afNegative
   af = [activationFunction[t] for t in tag]
   child = ann.ArbitraryNeuralNetwork(af)

   # set the weights of the net
   index = {tag[i] : i for i in range(len(tag))}
   for (target, source), w in weight.items():
      #t = tag.index(target)
      #s = tag.index(source)
      t = index[target]
      s = index[source]
      child.weight[t][s] = w

   return child

import collections
Candidate = collections.namedtuple('Candidate', ['ind','gen'])

generation = 0

# create a population of random nets
POP_SIZE = 64
MAX_STEPS = 32 * 32
myRandomNet = lambda: randomNet(random.randrange(8, 256))
population = [
   Candidate(myRandomNet(), generation)
   for _ in range(POP_SIZE)]

# enforcing at least some criterion on offspring
# rewards useful, long germ lines
N_NEURON_MIN = 8
while True:
   try:
      iParent = random.randrange(len(population))
      parent = population[iParent].ind
      child = decodeANN(parent, MAX_STEPS)

      if child.size < N_NEURON_MIN:
         population[iParent] = Candidate(
            myRandomNet(), 0)
      else:
         population[iParent] = Candidate(
            child, population[iParent].gen + 1)

      if False:
         # kill parents with cruddy children
         if child.size < N_NEURON_MIN:
            population[iParent] = Candidate(myRandomNet(), generation)
            print(".", end="")
         # otherwise propogate by replacing another individual
         else:
            iDead = random.randrange(len(population) - 1)
            if iDead >= iParent:
               iDead += 1
            population[iDead] = Candidate(child, population[iParent].gen)

      generation += 1
   except KeyboardInterrupt:
      populationSorted = sorted(
         population,
         key = lambda p: p.ind.size,
         reverse = True)
      print()
      fs = "{:4d}"
      print( " ".join(fs.format(p.gen) for p in populationSorted) )
      print( " ".join(fs.format(p.ind.size) for p in populationSorted) )
      input()

#net = randomNet(16)
#print(decodeANN(net).activationFunction)
