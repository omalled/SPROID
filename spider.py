import yaml
import copy
import math
import scipy as sp
import scipy.optimize as opt
import scipy.stats as stat
import numpy as np
import operator

def lognormpdf(x, mu, s):
	""" Calculate log-pdf of x, when x ~ N(mu,s) with s the covariance matrix"""
	tmp = -0.5 * (len(s) * math.log(2 * math.pi) + np.linalg.slogdet(s)[1])
	diff = x - mu
	if (sp.sparse.issparse(s)):
	    numerator = 0.5 * sp.sparse.linalg.spsolve(s, diff).T.dot(diff)
	else:
	    numerator = 0.5 * np.linalg.solve(s, diff).T.dot(diff)
	return tmp - numerator

def aic(maximumLikelihood, numParameters):
	""" Calculate the Akaike Information Criteria """
	return 2 * numParameters - 2 * math.log(maximumLikelihood)

def aicc(maximumLikelihood, numParameters, numSamples):
	""" Calculate the corrected Akaike Information Criteria """
	correction = 2 * numParameters * (numParameters + 1) / (numSamples - numParameters - 1)
	return aic(maximumLikelihood, numParameters) + correction

class Spider:
	trajectories = []
	def __init__(self, filename):
		stream = file(filename, 'r')
		data = yaml.load(stream)
		self.trajectories = map(Trajectory, data["Trajectories"])
	def getTrajectories(self):
		return self.trajectories
	def getTrajectory(self, i):
		return self.getTrajectories()[i]
	def getNumTrajectories(self):
		return len(self.trajectories)

class Trajectory:
	dim = 0
	times = []
	positions = []
	displacements = []
	dts = []
	zeroedPositions = []
	zeroedTimes = []
	def __init__(self, listTrajectory):
		lt = copy.deepcopy(listTrajectory)
		self.dim = len(lt[0]) - 1
		self.times = map(lambda x: x.pop(0), lt)
		self.positions = map(lambda x: map(lambda x: x.pop(0), lt), range(0, self.dim))
		self.displacements = map(lambda w: map(lambda x, y: y - x, w[:-1], w[1:]), self.positions)
		self.dts = map(lambda x, y: y - x, self.times[:-1], self.times[1:])
		self.zeroedPositions = map(lambda x: map(lambda y: y - x[0], x), self.positions)
		self.zeroedTimes = map(lambda t: t - self.times[0], self.times)
	def len(self):
		return len(self.times)
	def getDim(self):
		return dim
	def getTimes(self):
		return self.times
	def getPositions(self):
		return self.positions
	def getPositions1D(self, posIndex):
		return self.positions[posIndex]
	def getDisplacements(self):
		return self.displacements
	def getDisplacements1D(self, posIndex):
		return self.displacements[posIndex]
	def getDts(self):
		return self.dts
	#Returns a list containing tuples containing the AICC, the
	#maximum likelhood, and the maximum likelihood parameters for each
	#stochastic process in the list stochasticProcesses.
	#The analysis is done for the positions at posIndex
	def aicc1D(self, posIndex, stochasticProcesses):
		listTraj1D = map(lambda t, x: [t, x], self.times, self.positions[posIndex])
		traj1D = Trajectory(listTraj1D)
		mlList = map(lambda p: p.maximumLikelihood(traj1D), stochasticProcesses)
		#mlList is a list of lists containing the maximum likelihood and the maximum likelihood parameters
		return map(lambda x: (aicc(x[0], len(x[1]), len(self.times)), x[0], x[1]), mlList)
	def aic1D(self, posIndex, stochasticProcesses):
		listTraj1D = map(lambda t, x: [t, x], self.times, self.positions[posIndex])
		traj1D = Trajectory(listTraj1D)
		mlList = map(lambda p: p.maximumLikelihood(traj1D), stochasticProcesses)
		#mlList is a list of lists containing the maximum likelihood and the maximum likelihood parameters
		return map(lambda x: (aic(x[0], len(x[1])), x[0], x[1]), mlList)

class StochasticProcess:
	paramsMin = []
	paramsMax = []
	def __init__(self, paramsMin, paramsMax):
		self.paramsMin = paramsMin
		self.paramsMax = paramsMax
	def maximumLikelihood(self, trajectory):
		if trajectory.dim == 1:
			result = opt.minimize(lambda x: -self.transformedPDF(x, trajectory), map(lambda x, y: 0.5 * (x + y), self.paramsMin, self.paramsMax), method='TNC', bounds=map(lambda x, y: (x, y), self.paramsMin, self.paramsMax))
			return [self.pdf(result.x, trajectory), result.x]
		else:
			raise NotImplementedError("Only one dimensional trajectories are currently supported")
	#transformedPDF should be an increasing function of the PDF, e.g., log(pdf(params, trajectory))
	def transformedPDF(self, params, trajectory):
		raise NotImplementedError("This needs to be implemented in each subclass")
	def pdf(self, params, trajectory):
		if hasattr(self, 'logPDF'):
			return math.exp(self.logPDF(params, trajectory))
		else:
			raise NotImplementedError("This needs to be implemented in each subclass")

class BrownianMotion1D(StochasticProcess):
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def logPDF(self, params, trajectory):
		dxs = trajectory.getDisplacements1D(0)
		dts = trajectory.getDts()
		s = self.getSigma(params)
		sigmas = map(lambda x: math.sqrt(x) * s, dts)
		logpdfs = map(lambda dx, sigma: stat.norm.logpdf(dx, loc=0, scale=sigma), dxs, sigmas)
		sumlogpdfs = sum(logpdfs)
		return sumlogpdfs
	def getSigma(self, params):
		return params[0]

class FractionalBrownianMotion1D(StochasticProcess):
	def covarianceMatrix(self, params, trajectory):
		sigma = self.getSigma(params)
		sigma2 = sigma * sigma
		H = self.getHurstExponent(params)
		return np.matrix(map(lambda t: map(lambda s: 0.5 * sigma2 * (pow(abs(s), 2 * H) + pow(abs(t), 2 * H) - pow(abs(t - s), 2 * H)), trajectory.zeroedTimes[1:]), trajectory.zeroedTimes[1:]))
	def logPDF(self, params, trajectory):
		s = self.covarianceMatrix(params, trajectory)
		return lognormpdf(trajectory.zeroedPositions[0][1:], np.zeros(len(trajectory.zeroedPositions[0]) - 1), s)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def getSigma(self, params):
		return params[0]
	def getHurstExponent(self, params):
		return params[1]

def testCode():
	s = Spider('test.yaml')
	bm = BrownianMotion1D([0.1], [10.0])
	fbm = FractionalBrownianMotion1D([0.1, 0.00001], [10.0, 0.99999])
	print s.getTrajectories()
	for i, t in enumerate(s.getTrajectories()):
		print "trajectory " + str(i) + " (length: " + str(t.len()) + ")"
		print "\ttimes: " + str(t.getTimes())
		for j, p in enumerate(t.getPositions()):
			print "\tposition " + str(j) + ": " + str(p)
			print "\taicc: " + str(t.aicc1D(j, [bm, fbm]))
			print "\taic: " + str(t.aic1D(j, [bm, fbm]))

testCode()
