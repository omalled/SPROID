import random
import math
import matplotlib.pyplot as plt

class Polymer:
	def __init__(self, beta, K, N):
		"""Initialize a polymer with N monomers

		beta is equal to 1/(k_b T) where k_b is the Boltzmann constant and
		T is the temperature
		K is the force constant of the "springs" connecting the monomers
		(from Hooke's law)

		"""
		self.beta = beta
		self.K = K
		self.x = map(lambda x: math.sqrt(K * beta) * (x - .5 * N), range(0, N))
	def timeStep(self):
		N = len(self.x)
		for i in range(0, N):
			self.stepMC()
	def stepMC(self):
		i = random.randrange(0, len(self.x))
		lengthScale = math.sqrt(self.K * self.beta)
		#dx = random.gauss(0, lengthScale)
		dx = random.uniform(-lengthScale, lengthScale)
		dlogLikelihood = 0
		if i < len(self.x) - 1:
			dlogLikelihood -= -.5 * self.beta * self.K * dx * (2 * self.x[i + 1] - 2 * self.x[i] - dx)
		if i > 0:
			dlogLikelihood -= -.5 * self.beta * self.K * dx * (2 * self.x[i - 1] - 2 * self.x[i] - dx)
		if dlogLikelihood > 0 or math.exp(dlogLikelihood) > random.uniform(0, 1):
			self.x[i] += dx
		#print str(dlogLikelihood) + ": " + str(self.x)

def test():
	N = 200
	L = 129
	timeSteps = 2000
	polymers = map(lambda x: Polymer(1, 1, L), range(0, N))
	var = map(lambda x: 0, range(0, timeSteps))
	for i in range(0, timeSteps):
		for j in range(0, N):
			polymers[j].timeStep()
		#print "var: " + str(sum(map(lambda x: x.x[L / 2] * x.x[L / 2], polymers)) / N)
		var[i] = sum(map(lambda x: x.x[L / 2] * x.x[L / 2], polymers)) / N
		#print "mean: " + str(sum(map(lambda x: x.x[L / 2], polymers)) / N)
	plt.plot(var)
	plt.show()





