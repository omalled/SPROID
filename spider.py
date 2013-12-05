import yaml
import copy
import math
import scipy as sp
import scipy.optimize as opt
import scipy.stats as stat
import numpy as np
import operator
import ctypes
import multiprocessing
import subprocess
from StringIO import StringIO
import matplotlib.pyplot as plt

def subselect(lst, indices):
	return map(lambda x: x[1], filter(lambda y: y[0] in indices, enumerate(lst)))

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
	#print (numerator, denominator, np.linalg.slogdet(s)[1], len(s) * math.log(2 * math.pi))
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
	def __init__(self, filename, downsamples=[]):
		stream = file(filename, "r")
		data = yaml.load(stream)
		self.trajectories = map(lambda x: Trajectory(x, downsamples), data["Trajectories"])
	def getTrajectories(self):
		return self.trajectories
	def getTrajectory(self, i):
		return self.getTrajectories()[i]
	def getNumTrajectories(self):
		return len(self.trajectories)
	def aicOneTrajectory1D(self, trajIndex, posIndex):
		bm = BrownianMotion1D([1e-20], [1e-1], logParams=[True])
		bmres = self.getTrajectory(trajIndex).aic1D(posIndex, [bm])
		print bmres
		#bmd = BrownianMotionWithDrift1D([1e-20, -1], [1e-1, 1], params0=[bmres[0][0], 0.])
		bmd = BrownianMotionWithDrift1D([1e-20, -1], [1e-1, 1], logParams=[True, False])
		bmdres = self.getTrajectory(trajIndex).aic1D(posIndex, [bmd])
		print bmdres
		#fbm = FractionalBrownianMotion1D([1e-20, 0.1], [1e-1, 0.9], params0=[bmres[0][0], 0.5])
		fbm = FractionalBrownianMotion1D([1e-20, 0.1], [1e-1, 0.9], logParams=[True, False])
		fbmres = self.getTrajectory(trajIndex).aic1D(posIndex, [fbm])
		print fbmres
		#slm = SymmetricLevyMotion1D([0.5, 1e-20], [1.999, 1e-1], params0=[1.999, bmres[0][0] / math.sqrt(2)])
		slm = SymmetricLevyMotion1D([0.5, 1e-20], [1.999, 1e-1], logParams=[False, True])
		slmres = self.getTrajectory(trajIndex).aic1D(posIndex, [slm])
		print slmres
		#slmd = SymmetricLevyMotionWithDrift1D([0.5, 1e-20, -1.], [1.999, 1e-1, 1.], params0=[slmres[0][0], slmres[0][1], 0.])
		slmd = SymmetricLevyMotionWithDrift1D([0.5, 1e-20, -1.], [1.999, 1e-1, 1.], logParams=[False, True, False])
		slmdres = self.getTrajectory(trajIndex).aic1D(posIndex, [slmd])
		print slmdres
		return [bmres, bmdres, fbmres, slmres, slmdres]

class Trajectory:
	dim = 0
	length = 0
	times = []
	positions = []
	displacements = []
	dts = []
	zeroedPositions = []
	zeroedTimes = []
	def __init__(self, listTrajectory, downsamples=[]):
		"""Creates a trajectory from listTrajectory.

		listTrajectory should be a list with each entry being of the
		form [t, x, y, ...]
		"""
		if downsamples == []:
			lt = copy.deepcopy(listTrajectory)
		else:
			lt = subselect(listTrajectory, map(lambda x: int(round(x * len(listTrajectory) / (downsamples - 1.))), range(0, downsamples)))
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
	def getTrajectory1D(self, posIndex):
		listTraj1D = map(lambda t, x: [t, x], self.times, self.positions[posIndex])
		traj1D = Trajectory(listTraj1D)
		return traj1D
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
	def plot1D(self, posIndex, xlabel='Time', ylabel='Position', show=True):
		plt.plot(self.getTimes(), self.getPositions1D(posIndex))
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if show:
			plt.show()

class StochasticProcess:
	paramsMin = []
	paramsMax = []
	params0 = []
	logParams = []
	def __init__(self, paramsMin, paramsMax, params0=[], logParams=[]):
		self.paramsMin = paramsMin
		self.paramsMax = paramsMax
		self.params0 = params0
		self.logParams = logParams
		if self.logParams == []:
			self.logParams = map(lambda x: False, paramsMin)
		for i in range(0, len(paramsMin)):
			if self.logParams[i]:
				self.paramsMin[i] = math.log10(paramsMin[i])
				self.paramsMax[i] = math.log10(paramsMax[i])
			else:
				self.paramsMin[i] = paramsMin[i]
				self.paramsMax[i] = paramsMax[i]
	def transformParam(self, params, i):
		if self.logParams[i]:
			return math.pow(10, params[i])
		else:
			return params[i]
	def maximumLogLikelihood(self, trajectory):
		if trajectory.dim == 1:
			if self.params0 == []:
				p0 = map(lambda x, y: 0.5 * (x + y), self.paramsMin, self.paramsMax)
			else:
				p0 = self.params0
			#Note that using TNC can cause the function to evaluate outside the bounded range (when it tries to evaluate derivatives, so give a little cushion, if necessary)
			#result = opt.minimize(lambda x: -self.transformedPDF(x, trajectory), p0, method="TNC", bounds=map(lambda x, y: (x, y), self.paramsMin, self.paramsMax), options={'disp': True})
			#return [self.logPDF(result.x, trajectory), result.x]
			x = opt.fmin_tnc(lambda x: -self.transformedPDF(x, trajectory), p0, maxfun=1000, bounds=map(lambda x, y: (x, y), self.paramsMin, self.paramsMax), disp=0, approx_grad=True)[0]
			return [self.logPDF(x, trajectory), x]
			#x = opt.anneal(lambda x: -self.transformedPDF(x, trajectory), p0, maxeval=1000, lower=self.paramsMin, upper=self.paramsMax, disp=0)[0]
			#return [self.logPDF(x, trajectory), x]
			#x = opt.basin_hopping(lambda x: -self.transformedPDF(x, trajectory), p0, maxeval=1000, lower=self.paramsMin, upper=self.paramsMax, disp=0)[0]
			#return [self.logPDF(x, trajectory), x]
			#x = opt.fmin_l_bfgs_b(lambda x: -self.transformedPDF(x, trajectory), p0, maxfun=1000, bounds=map(lambda x, y: (x, y), self.paramsMin, self.paramsMax), disp=0, approx_grad=True)[0]
			#return [self.logPDF(x, trajectory), x]
			#result = opt.brute(lambda x: -self.transformedPDF(x, trajectory), ranges=map(lambda x, y: (x, y), self.paramsMin, self.paramsMax), disp=False)
			#return [self.logPDF(result, trajectory), result]
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
		return self.transformParam(params, 0)
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
		return self.transformParam(params, 1)

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
		return self.transformParam(params, 0)
	def getHurstExponent(self, params):
		return self.transformParam(params, 1)

#for lm, we have to import a C library to compute the log-likelihood
lmlib = np.ctypeslib.load_library('lm', '.')
lmLogLikelihood = lmlib.log_likelihood
lmLogLikelihood.restype = ctypes.c_double
lmLogLikelihood.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), ctypes.c_int]
class LevyMotion1D(StochasticProcess):
	def logPDF(self, params, trajectory):
		dxs = trajectory.getDisplacements1D(0)
		dts = trajectory.getDts()
		return lmLogLikelihood(self.getAlpha(params), self.getBeta(params), 0., self.getLambda(params), np.array(dts, dtype=np.float64), np.array(dxs, dtype=np.float64), trajectory.len() - 1)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def getAlpha(self, params):
		return self.transformParam(params, 0)
	def getBeta(self, params):
		return self.transformParam(params, 1)
	def getLambda(self, params):
		return self.transformParam(params, 2)

#for lm, we have to import a C library to compute the log-likelihood
symlmLogLikelihood = lmlib.sym_log_likelihood
symlmLogLikelihood.restype = ctypes.c_double
symlmLogLikelihood.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), ctypes.c_int]
class SymmetricLevyMotion1D(LevyMotion1D):
	def logPDF(self, params, trajectory):
		dxs = trajectory.getDisplacements1D(0)
		dts = trajectory.getDts()
		return symlmLogLikelihood(self.getAlpha(params), 0., self.getLambda(params), np.array(dts, dtype=np.float64), np.array(dxs, dtype=np.float64), trajectory.len() - 1)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def getAlpha(self, params):
		return self.transformParam(params, 0)
	def getBeta(self, params):
		return 0.
	def getLambda(self, params):
		return self.transformParam(params, 1)

class SymmetricLevyMotionWithDrift1D(SymmetricLevyMotion1D):
	def logPDF(self, params, trajectory):
		dxs = trajectory.getDisplacements1D(0)
		dts = trajectory.getDts()
		return symlmLogLikelihood(self.getAlpha(params), self.getVelocity(params), self.getLambda(params), np.array(dts, dtype=np.float64), np.array(dxs, dtype=np.float64), trajectory.len() - 1)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def getVelocity(self, params):
		return self.transformParam(params, 2)

#split to yaml converts the "tracks.txt" file output by I. Jankovic's split
#code into spider yaml files
def splitToYaml(filename):
	infile = open(filename, "r")
	data = infile.read()
	infile.close()
	bigArray = np.genfromtxt(StringIO(data), delimiter=",")
	numTrajectories = int(bigArray[-1][-1])
	listTrajectories = map(lambda x: [], range(0, numTrajectories))
	for line in bigArray[1:]:
		x = np.asscalar(line[0])
		y = np.asscalar(line[1])
		t = np.asscalar(line[-2])
		trajNum = int(line[-1])
		listTrajectories[trajNum-1].append([t, x, y])
	#listTrajectories = [[[0, 0], [1, 1]], [[0, 0], [1, 1]]]
	data = {"Trajectories": listTrajectories}
	outfile = open(filename + ".yaml", "w")
	yaml.dump(data, outfile)
	outfile.close()

#split to yaml converts the "tracks.txt" file output by I. Jankovic's split
#code into spider yaml files
def fractureToYaml(filename):
	infile = open(filename, "r")
	data = infile.read()
	infile.close()
	bigArray = np.genfromtxt(StringIO(data), delimiter=" ")
	numTrajectories = 1
	listTrajectories = map(lambda x: [], range(0, numTrajectories))
	for line in bigArray:
		x = np.asscalar(line[2])
		y = np.asscalar(line[3])
		z = np.asscalar(line[4])
		t = np.asscalar(line[-1])
		listTrajectories[0].append([t, x, y, z])
	#listTrajectories = [[[0, 0], [1, 1]], [[0, 0], [1, 1]]]
	data = {"Trajectories": listTrajectories}
	outfile = open(filename + ".yaml", "w")
	yaml.dump(data, outfile)
	outfile.close()

def convertNataliiasData():
	for i in range(1, 2784):
		fractureToYaml("fracture/traject_" + str(i))

def testCode():
	#s = Spider("traject_1.yaml")
	s = Spider("test.yaml")
	bm = BrownianMotion1D([0.1], [10.0])
	bmd = BrownianMotionWithDrift1D([0.1, -1], [10.0, 1])
	fbm = FractionalBrownianMotion1D([0.1, 0.1], [10.0, 0.9])
	lm = LevyMotion1D([0.5, -1., 0.1], [2.0, 1., 10.0])
	slm = SymmetricLevyMotion1D([0.5, 0.1], [1.99, 10.0])
	slmd = SymmetricLevyMotionWithDrift1D([0.5, 0.1, 0.1], [1.99, 10.0, 10.])
	traj = s.getTrajectory(0)
	#traj = BrownianMotion1D.getTrajectory([2.345], range(1, 100))
	print "aicc: " + str(traj.aicc1D(0, [bm, bmd, fbm, slm, slmd]))

#testCode()
bm = BrownianMotion1D([0.1], [10.0])
bmd = BrownianMotionWithDrift1D([0.1, -10.], [10.0, 10.])
fbm = FractionalBrownianMotion1D([0.1, 0.1], [10.0, 0.9])
slm = SymmetricLevyMotion1D([0.5, 0.1], [1.999, 10.0])
slmd = SymmetricLevyMotionWithDrift1D([0.5, 0.1, -10.], [1.999, 10.0, 10.])
defaultSPs = [bm, bmd, fbm, slm, slmd]

slowbm = BrownianMotion1D([1e-20], [1e-1], logParams=[True])
slowbmd = BrownianMotionWithDrift1D([1e-20, -1e-3], [1e-1, 1e-3], logParams=[True, False])
slowfbm = FractionalBrownianMotion1D([1e-20, 0.1], [1e-1, 0.9], logParams=[True, False])
slowslm = SymmetricLevyMotion1D([0.5, 1e-20], [1.999, 1e-1], logParams=[False, True])
slowslmd = SymmetricLevyMotionWithDrift1D([0.5, 1e-20, -1], [1.999, 1e-1, 1.])
slowSPs = [slowbm, slowbmd, slowfbm, slowslm, slowslmd]

def testFracture():
	s = Spider("traject_1.yaml", downsamples=100)
	traj = s.getTrajectory(0)
	results = traj.aic1D(0, [bm, bmd, fbm, slm])
	return results

def runFracture(filenum, posIndex=0):
	s = Spider("fracture/traject_" + str(filenum) + ".yaml", downsamples=50)
	traj = s.getTrajectory(0)
	traj = traj.getTrajectory1D(posIndex)
	results = traj.aic1D(0, [bm, bmd, fbm, slm])
	minaic = results[0][0]
	for i in range(1, 4):
		if results[i][0] < minaic:
			minaic = results[i][0]
	outfile = open("fracture/traject_" + str(filenum) + ".results" + str(posIndex), "w")
	for i in range(0, 4):
		if results[i][0] == minaic:
			if i == 0:
				outfile.write("BM\n")
			if i == 1:
				outfile.write("BMD\n")
			if i == 2:
				outfile.write("FBM\n")
			if i == 3:
				outfile.write("SLM\n")
	outfile.write("BM " + str(results[0][0]) + "\n")
	outfile.write(str(results[0][2][0]) + "\n")
	outfile.write("BMD " + str(results[1][0]) + "\n")
	outfile.write(str(results[1][2][0]) + " " + str(results[1][2][1]) + "\n")
	outfile.write("FBM " + str(results[2][0]) + "\n")
	outfile.write(str(results[2][2][0]) + " " + str(results[2][2][1]) + "\n")
	outfile.write("SLM " + str(results[3][0]) + "\n")
	outfile.write(str(results[3][2][0]) + " " + str(results[3][2][1]) + "\n")
	outfile.close()

def runFracture1(i):
	runFracture(i, 1)

def runFracture2(i):
	runFracture(i, 2)

def runFractures():
	n = 2784
	#n = 6
	pool = multiprocessing.Pool(4)
	pool.map(runFracture, range(1, n))
	pool.map(runFracture1, range(1, n))
	pool.map(runFracture2, range(1, n))
	#pool.map(runFracture, range(1, n), map(lambda x: 0, range(1, n)))
	#pool.map(runFracture, range(1, n), map(lambda x: 1, range(1, n)))
	#pool.map(runFracture, range(1, n), map(lambda x: 2, range(1, n)))

#runFractures()

def runSplit():
	n = 100
	pool = multiprocessing.Pool(4)
	pool.map(runSplitSupport, range(0, n))

def runSplitSupport(k):
	s = Spider("split/tracks.txt.yaml", downsamples=50)
	traj2D = s.getTrajectory(k)
	if traj2D.getPositions1D(0)[-1] > 0:
		for j in range(0, 2):
			traj = traj2D.getTrajectory1D(j)
			results = traj.aic1D(0, [slowbm, slowbmd, slowfbm, slowslm])
			minaic = results[0][0]
			for i in range(1, 4):
				if results[i][0] < minaic:
					minaic = results[i][0]
			outfile = open("split/traject_" + str(k) + ".results" + str(j), "w")
			for i in range(0, 4):
				if results[i][0] == minaic:
					if i == 0:
						outfile.write("BM\n")
					if i == 1:
						outfile.write("BMD\n")
					if i == 2:
						outfile.write("FBM\n")
					if i == 3:
						outfile.write("SLM\n")
			outfile.write("BM " + str(results[0][0]) + "\n")
			outfile.write(str(results[0][2][0]) + "\n")
			outfile.write("BMD " + str(results[1][0]) + "\n")
			outfile.write(str(results[1][2][0]) + " " + str(results[1][2][1]) + "\n")
			outfile.write("FBM " + str(results[2][0]) + "\n")
			outfile.write(str(results[2][2][0]) + " " + str(results[2][2][1]) + "\n")
			outfile.write("SLM " + str(results[3][0]) + "\n")
			outfile.write(str(results[3][2][0]) + " " + str(results[3][2][1]) + "\n")
			outfile.close()

#runSplit()

def summarizeResults(folder):
	lsOutput = subprocess.check_output("ls -1 " + folder + "/*.results2", shell=True)
	filenames = lsOutput.split('\n')
	winnerCount = {}
	winnerCount["BM"] = 0
	winnerCount["BMD"] = 0
	winnerCount["FBM"] = 0
	winnerCount["SLM"] = 0
	mlParams = [[], [], [], []]
	aics = [[], [], [], []]
	for filename in filenames[0:-1]:
		#f = open(folder + "/" + filename, "r")
		f = open(filename, "r")
		lines = f.readlines()
		f.close()
		winnerCount[lines[0].rstrip()] += 1
		for i in range(0, 4):
			chunks = lines[1 + 2 * i].split(" ")
			aics[i].append(float(chunks[1]))
			chunks = lines[2 + 2 * i].split(" ")
			mlParams[i].append(map(float, chunks))
	print winnerCount
	print map(lambda x: sum(x) / float(len(x)), aics)
	print map(lambda x: map(lambda y: sum(y) / float(len(y)), zip(*x)), mlParams)
