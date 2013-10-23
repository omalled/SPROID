import yaml
import copy
import math
import scipy as sp
import scipy.optimize as opt
import scipy.stats as stat
import numpy as np
import operator
import ctypes

def diff(l):
	return map(lambda x, y: y - x, l[:-1], l[1:])

def lognormpdf(x, mu, s):
	""" Calculate log-pdf of x, when x ~ N(mu,s) with s the covariance matrix"""
	denominator = 0.5 * (len(s) * math.log(2 * math.pi) + np.linalg.slogdet(s)[1])
	diff = x - mu
	if (sp.sparse.issparse(s)):
	    numerator = -0.5 * sp.sparse.linalg.spsolve(s, diff).T.dot(diff)
	else:
		numerator = -0.5 * np.linalg.solve(s, diff).T.dot(diff)
	print (numerator, denominator, np.linalg.slogdet(s)[1], len(s) * math.log(2 * math.pi))
	return numerator - denominator

def aic(maximumLogLikelihood, numParameters):
	""" Calculate the Akaike Information Criteria """
	return 2 * numParameters - 2 * maximumLogLikelihood

def aicc(maximumLogLikelihood, numParameters, numSamples):
	""" Calculate the corrected Akaike Information Criteria """
	correction = 2. * numParameters * (numParameters + 1.) / (numSamples - numParameters - 1.)
	return aic(maximumLogLikelihood, numParameters) + correction

class Spider:
	trajectories = []
	def __init__(self, filename):
		stream = file(filename, "r")
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
	length = 0
	times = []
	positions = []
	displacements = []
	dts = []
	zeroedPositions = []
	zeroedTimes = []
	def __init__(self, listTrajectory):
		"""Creates a trajectory from listTrajectory.

		listTrajectory should be a list with each entry being of the
		form [t, x, y, ...]
		"""
		lt = copy.deepcopy(listTrajectory)
		self.dim = len(lt[0]) - 1
		self.times = map(lambda x: x.pop(0), lt)
		self.positions = map(lambda x: map(lambda x: x.pop(0), lt), range(0, self.dim))
		self.displacements = map(lambda w: map(lambda x, y: y - x, w[:-1], w[1:]), self.positions)
		self.dts = diff(self.times)
		self.zeroedPositions = map(lambda x: map(lambda y: y - x[0], x), self.positions)
		self.zeroedTimes = map(lambda t: t - self.times[0], self.times)
		self.length = len(self.times)
	def len(self):
		return self.length
	def getDim(self):
		return self.dim
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
		mlList = map(lambda p: p.maximumLogLikelihood(traj1D), stochasticProcesses)
		#mlList is a list of lists containing the maximum likelihood and the maximum likelihood parameters
		return map(lambda x: (aicc(x[0], len(x[1]), len(self.times)), x[0], x[1]), mlList)
	def aic1D(self, posIndex, stochasticProcesses):
		listTraj1D = map(lambda t, x: [t, x], self.times, self.positions[posIndex])
		traj1D = Trajectory(listTraj1D)
		mlList = map(lambda p: p.maximumLogLikelihood(traj1D), stochasticProcesses)
		#mlList is a list of lists containing the maximum likelihood and the maximum likelihood parameters
		return map(lambda x: (aic(x[0], len(x[1])), x[0], x[1]), mlList)

class StochasticProcess:
	paramsMin = []
	paramsMax = []
	def __init__(self, paramsMin, paramsMax):
		self.paramsMin = paramsMin
		self.paramsMax = paramsMax
	def maximumLogLikelihood(self, trajectory):
		if trajectory.dim == 1:
			result = opt.minimize(lambda x: -self.transformedPDF(x, trajectory), map(lambda x, y: 0.5 * (x + y), self.paramsMin, self.paramsMax), method="TNC", bounds=map(lambda x, y: (x, y), self.paramsMin, self.paramsMax))
			return [self.logPDF(result.x, trajectory), result.x]
		else:
			raise NotImplementedError("Only one dimensional trajectories are currently supported")
	#transformedPDF should be an increasing function of the PDF, e.g., log(pdf(params, trajectory))
	def transformedPDF(self, params, trajectory):
		raise NotImplementedError("This needs to be implemented in each subclass")
	def pdf(self, params, trajectory):
		return math.exp(self.logPDF(params, trajectory))
	def logPDF(self, params, trajectory):
		raise NotImplementedError("This needs to be implemented in each subclass")
	@staticmethod
	def getTrajectory(params, times):
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
	@staticmethod
	def getTrajectory(params, times):
		"""Generates a trajectory of a Brownian Motion in 1 dimension.

		times should all be non-negative and increasing
		"""
		previousT = 0
		sigma = params[0]
		x = [stat.norm.rvs(loc=0, scale=sigma*times[0])]
		for dt in diff(times):
			x.append(stat.norm.rvs(loc=x[-1], scale=sigma * math.sqrt(dt)))
		return Trajectory(map(lambda t, pos: [t, pos], times, x))

class BrownianMotionWithDrift1D(BrownianMotion1D):
	def logPDF(self, params, trajectory):
		dxs = trajectory.getDisplacements1D(0)
		dts = trajectory.getDts()
		s = self.getSigma(params)
		v = self.getVelocity(params)
		sigmas = map(lambda x: math.sqrt(x) * s, dts)
		vdts = map(lambda x: v * x, dts)
		logpdfs = map(lambda dx, sigma, vdt: stat.norm.logpdf(dx, vdt, scale=sigma), dxs, sigmas, vdts)
		sumlogpdfs = sum(logpdfs)
		return sumlogpdfs
	def getVelocity(self, params):
		return params[1]

#for fbm, we have to import a C library to compute the log-likelihood
fbmlib = np.ctypeslib.load_library('fbm', '.')
fbmLogLikelihood = fbmlib.log_likelihood
fbmLogLikelihood.restype = ctypes.c_double
fbmLogLikelihood.argtypes = [ctypes.c_double, ctypes.c_double, np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), ctypes.c_int]
class FractionalBrownianMotion1D(StochasticProcess):
	def covarianceMatrix(self, params, trajectory):
		sigma = self.getSigma(params)
		sigma2 = sigma * sigma
		H = self.getHurstExponent(params)
		return np.array(map(lambda t: map(lambda s: 0.5 * sigma2 * (pow(abs(s), 2 * H) + pow(abs(t), 2 * H) - pow(abs(t - s), 2 * H)), trajectory.zeroedTimes[1:]), trajectory.zeroedTimes[1:]))
	def logPDF(self, params, trajectory):
		return fbmLogLikelihood(self.getSigma(params), self.getHurstExponent(params), np.array(trajectory.zeroedTimes[1:], dtype=np.float64), np.array(trajectory.zeroedPositions[0][1:], dtype=np.float64), trajectory.len() - 1)
	def pylogPDF(self, params, trajectory):
		s = self.covarianceMatrix(params, trajectory)
		return lognormpdf(trajectory.zeroedPositions[0][1:], np.zeros(len(trajectory.zeroedPositions[0]) - 1), s)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def getSigma(self, params):
		return params[0]
	def getHurstExponent(self, params):
		return params[1]

def testCode():
	s = Spider("traject_1.yaml")
	#s = Spider("test.yaml")
	bm = BrownianMotion1D([0.1], [10.0])
	bmd = BrownianMotionWithDrift1D([0.1, -1], [10.0, 1])
	fbm = FractionalBrownianMotion1D([0.1, 0.1], [10.0, 0.9])
	traj = s.getTrajectory(0)
	#for i in range(0, 20):
		#traj = BrownianMotion1D.getTrajectory([2.345], range(1, 100))
		#traj = Trajectory([[1, 1], [2, 2], [3, 1], [4, 0]])
		#print "bm: " + str(bm.logPDF([10.], traj))
		#pos = traj.getPositions1D(0)
		#print "v: " + str((pos[-1] - pos[0]) / (len(pos) - 1)) + ", sigma: " + str(math.sqrt(sum(map(lambda x: x * x, diff(pos))) / (len(pos) - 2)))
		#print "aicc: " + str(traj.aicc1D(0, [fbm]))
	traj = BrownianMotion1D.getTrajectory([2.345], range(1, 300))
	print "aicc: " + str(traj.aicc1D(0, [fbm]))
	#print fbm.logPDF([ 1.40676782,  0.80235562], traj)
	#print fbm.pylogPDF([ 1.40676782,  0.80235562], traj)

	"""
	for i, t in enumerate(s.getTrajectories()):
		print "trajectory " + str(i) + " (length: " + str(t.len()) + ")"
		#print "\ttimes: " + str(t.getTimes())
		for j, p in enumerate(t.getPositions()):
			#print "\tposition " + str(j) + ": " + str(p)
			print "\taicc: " + str(t.aicc1D(j, [bm, bmd]))
			print "\taic: " + str(t.aic1D(j, [bm, bmd]))
			#print "\taicc: " + str(t.aicc1D(j, [bm]))
			#print "\taic: " + str(t.aic1D(j, [bm, fbm]))
			"""

testCode()
