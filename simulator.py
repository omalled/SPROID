import random
import math
import matplotlib.pyplot as plt
import spider
import multiprocessing
import itertools

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
		self.N = N
		self.x = map(lambda x: math.sqrt(K * beta) * (x - .5 * N), range(0, N))
	def timeStep(self):
		for i in range(0, self.N):
			self.stepMC()
	def stepMC(self):
		i = random.randrange(0, self.N)
		lengthScale = math.sqrt(self.K * self.beta)
		#dx = random.gauss(0, lengthScale)
		dx = random.uniform(-lengthScale, lengthScale)
		dlogLikelihood = 0
		if i < self.N - 1:
			dlogLikelihood -= -.5 * self.beta * self.K * dx * (2 * self.x[i + 1] - 2 * self.x[i] - dx)
		if i > 0:
			dlogLikelihood -= -.5 * self.beta * self.K * dx * (2 * self.x[i - 1] - 2 * self.x[i] - dx)
		if dlogLikelihood > 0 or math.exp(dlogLikelihood) > random.uniform(0, 1):
			self.x[i] += dx
		#print str(dlogLikelihood) + ": " + str(self.x)
	def getCentralMonomerPosition(self):
		return self.x[self.N / 2]
	def burnIn(self, numSteps):
		return
	@staticmethod
	def getTrajectory(params, times, numTrajs=1, numProcs=4):
		"""Gets trajectories of the central monomer.

		times should be a list of integers
		"""
		if numTrajs == 1:
			polymer = Polymer(*params)
			polymer.burnIn(polymer.N * polymer.N)
			trajectory = []
			for j in range(0, len(times) - 1):
				trajectory.append([times[j], polymer.getCentralMonomerPosition()])
				dt = times[j + 1] - times[j]
				for k in range(0, dt):
					polymer.timeStep()
			trajectory.append([times[-1], polymer.getCentralMonomerPosition()])
			return [spider.Trajectory(trajectory)]
		else:
			pool = multiprocessing.Pool(numProcs)
			a =  pool.map(polymerGetTrajectoryStar, itertools.izip(itertools.repeat(params, numTrajs), itertools.repeat(times, numTrajs)))
			return a
		"""
		trajectories = [[] for i in range(0, numTrajs)]
		for i in range(0, numTrajs):
			polymer = Polymer(*params)
			polymer.burnIn(polymer.N * polymer.N)
			for j in range(0, len(times) - 1):
				trajectories[i].append([times[j], polymer.getCentralMonomerPosition()])
				dt = times[j + 1] - times[j]
				for k in range(0, dt):
					polymer.timeStep()
			trajectories[i].append([times[-1], polymer.getCentralMonomerPosition()])
		return map(spider.Trajectory, trajectories)
		"""
	@staticmethod
	def testShort(params=[1, 1, 129], numTrajs=100, plot=True, numProcs=4):
		trajs = Polymer.getTrajectory(params, range(0, int(1e3), int(1e1)), numTrajs=numTrajs, numProcs=numProcs)
		s = spider.Spider(trajectories=trajs)
		s.getResults(plot=plot)
	@staticmethod
	def testLong(params=[1, 1, 33], numTrajs=100, plot=True, numProcs=4):
		trajs = Polymer.getTrajectory(params, range(0, int(1e6), int(1e4)), numTrajs=numTrajs, numProcs=numProcs)
		bm = spider.BrownianMotion1D([0.00001], [1.0])
		bmplc = spider.StochasticProcessWithNonlinearClock(spider.BrownianMotion1D, spider.power_law_clock, [0.00001, 0.5], [10.0, 1.5])
		fbm = spider.FractionalBrownianMotion1D([0.00001, 0.1], [1.0, 0.9])
		slm = spider.SymmetricLevyMotion1D([0.5, 0.00001], [1.999, 1.0])
		s = spider.Spider(trajectories=trajs)
		s.dump("polymerLong.yaml")
		s.getResults(plot=plot, sps=[bm, fbm, slm, bmplc])

def polymerGetTrajectoryStar(p):
	return Polymer.getTrajectory(p[0], p[1], numTrajs=1, numProcs=1)[0]

def test():
	N = 200
	L = 129
	timeSteps = 20000
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

