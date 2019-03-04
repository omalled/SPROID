"""
© (or copyright) 2014. Triad National Security, LLC. All rights reserved.
 
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.
 
All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

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
import levy
import itertools

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

class Sproid:
	trajectories = []
	def __init__(self, filename=[], downsamples=[], trajectories=[]):
		self.trajectories = trajectories
		if filename != []:
			stream = file(filename, "r")
			data = yaml.load(stream)
			self.trajectories.extend(map(lambda x: Trajectory(x, downsamples), data["Trajectories"]))
	def dump(self, filename):
		listTrajectories = []
		for i in range(0, len(self.trajectories)):
			listTrajectory = map(lambda x: [x], self.trajectories[i].getTimes())
			for j in range(0, self.trajectories[i].getDim()):
				for k in range(0, len(listTrajectory)):
					listTrajectory[k].append(self.trajectories[i].getPositions1D(j)[k])
			listTrajectories.append(listTrajectory)
		data = {"Trajectories": listTrajectories}
		outfile = open(filename, "w")
		outfile.write(yaml.dump(data))
		outfile.close()
	def getTrajectories(self):
		return self.trajectories
	def getTrajectory(self, i):
		return self.getTrajectories()[i]
	def getNumTrajectories(self):
		return len(self.trajectories)
	def aicOneTrajectory1D(self, trajIndex, posIndex, sps=[]):
		if sps == []:
			sps = defaultSPs
		results = []
		for sp in sps:
			result = self.getTrajectory(trajIndex).aic1D(posIndex, [sp])[0]
			results.append(result)
		return results
	def getResults(self, num_procs=4, plot=True, sps=[], posIndex=0):
		numTrajs = len(self.trajectories)
		if sps == []:
			sps = [bm, fbm, slm, bmplc]
		param_results = [[] for sp in sps]
		pool = multiprocessing.Pool(num_procs)
		#results = pool.map(self.aicOneTrajectory1DStar, zip([self for i in range(0, numTrajs)], range(0, numTrajs), [0 for i in range(0, numTrajs)], [sps for i in range(0, numTrajs)]))
		results = pool.map(sproidaicOneTrajectory1DStar, itertools.izip(itertools.repeat(self), range(0, numTrajs), itertools.repeat(posIndex), itertools.repeat(sps)))
		#results = map(sproidaicOneTrajectory1DStar, itertools.izip(itertools.repeat(self), range(0, numTrajs), itertools.repeat(0), itertools.repeat(sps)))
		summarizeResultsList(results, sps, plot=plot)
	@staticmethod
	def runTests(numTrajs=100):
		print "Testing with Brownian motion"
		BrownianMotion1D.test(plot=False, numTrajs=numTrajs)
		print "Testing with fractional Brownian motion"
		FractionalBrownianMotion1D.test(plot=False, numTrajs=numTrajs)
		print "Testing with symmetric Levy motion"
		SymmetricLevyMotion1D.test(plot=False, numTrajs=numTrajs)
		print "Testing with Brownian motion with a power-law clock"
		StochasticProcessWithNonlinearClock.test(plot=False, numTrajs=numTrajs)

def sproidaicOneTrajectory1DStar(args):
	return Sproid.aicOneTrajectory1D(*args)

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
	def getNumParams(self):
		raise NotImplementedError("This needs to be implemented in each subclass")
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
	def getTrajectory(params, times, numTrajs=1):
		raise NotImplementedError("This needs to be implemented in each subclass")
	def name(self):
		raise NotImplementedError("This needs to be implemented in each subclass")

class StochasticProcessWithNonlinearClock(StochasticProcess):
	def __init__(self, parentProcessClass, clock, paramsMin, paramsMax, params0=[], logParams=[]):
		"""Constructs a StochasticProcess with a nonlinear clock based on the parent process

		If the parent_process has N parameters, then the first N elements of paramsMin and
		paramsMax should correspond to those parameters the remaining parameters are clock
		parameters.

		"""
		StochasticProcess.__init__(self, paramsMin, paramsMax, params0, logParams)
		self.clock = clock
		self.parentProcess = parentProcessClass(paramsMin, paramsMax, params0, logParams)
		self.parentProcessClass = parentProcessClass
	def name(self):
		return self.parentProcess.name() + " with a nonlinear clock"
	def getNumParams(self):
		return len(paramsMin)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def logPDF(self, params, trajectory):
		clockTimes = map(lambda t: self.clock(t, params, self.parentProcess.getNumParams()), trajectory.getTimes())
		positions = trajectory.getPositions1D(0)
		timeChangedTrajectory = Trajectory(map(list, zip(clockTimes, positions)))
		return self.parentProcess.logPDF(params, timeChangedTrajectory)
	@staticmethod
	def getTrajectory(params, times, numTrajs=1, clock=lambda t, p, i: math.pow(t, p[i]), parentProcessClass=[]):
		"""Generates a trajectory of a stochastic process with a nonlinear clock

		times should all be non-negative and increasing
		"""
		if parentProcessClass == []:
			print "You must specify a parent class"
		parentProcess = parentProcessClass(params, params)
		clockTimes = map(lambda t: clock(t, params, parentProcess.getNumParams()), times)
		parentTrajectories = parentProcessClass.getTrajectory(params, clockTimes, numTrajs)
		nlcTrajectories = map(lambda x: Trajectory(map(list, zip(times, *x))), map(lambda y: y.getPositions(), parentTrajectories))
		return nlcTrajectories
	@staticmethod
	def test(params=[1., 0.5], numTrajs=100, plot=True):
		trajs = StochasticProcessWithNonlinearClock.getTrajectory(params, np.arange(0, 10, .1), numTrajs=numTrajs, parentProcessClass=BrownianMotion1D)
		s = Sproid(trajectories=trajs)
		s.getResults(plot=plot)

class BrownianMotion1D(StochasticProcess):
	def name(self):
		return "Brownian motion 1D"
	def getNumParams(self):
		return 1
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
	def getTrajectory(params, times, numTrajs=1):
		"""Generates a trajectory of a Brownian Motion in 1 dimension.

		times should all be non-negative and increasing
		"""
		if numTrajs > 1:
			return map(lambda x: BrownianMotion1D.getTrajectory(params, times, numTrajs=1)[0], range(0, numTrajs))
		else:
			previousT = 0
			sigma = params[0]
			x = [stat.norm.rvs(loc=0, scale=sigma*times[0])]
			for dt in diff(times):
				x.append(stat.norm.rvs(loc=x[-1], scale=sigma * math.sqrt(dt)))
			return [Trajectory(map(lambda t, pos: [t, pos], times, x))]
	@staticmethod
	def test(params=[1.], numTrajs=100, plot=True):
		trajs = BrownianMotion1D.getTrajectory(params, np.arange(0, 10, .1), numTrajs=numTrajs)
		s = Sproid(trajectories=trajs)
		s.getResults(plot=plot)

class BrownianMotionWithDrift1D(BrownianMotion1D):
	def name(self):
		return "Brownian motion with drift 1D"
	def getNumParams(self):
		return 2
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
	def name(self):
		return "Fractional Brownian motion 1D"
	def getNumParams(self):
		return 2
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
	@staticmethod
	def getTrajectory(params, times, numTrajs=1):
		"""Generates a trajectory of a fractional Brownian motion in 1 dimension.

		times should all be non-negative and increasing
		"""
		t0 = times[0]
		previousT = 0
		sigma = params[0]
		H = params[1]
		if t0 == 0:
			chol_covar = FractionalBrownianMotion1D.getCholCovar(params, times[1:])
		else:
			chol_covar = FractionalBrownianMotion1D.getCholCovar(params, times)
		trajs = []
		for i in range(0, numTrajs):
			if t0 == 0:
				trajs.append(np.array([0.] + list(chol_covar.dot(np.array(map(lambda x: stat.norm.rvs(loc=0, scale=1), range(0, len(times[1:]))))))))
			else:
				trajs.append(chol_covar.dot(np.array(map(lambda x: stat.norm.rvs(loc=0, scale=1), range(0, len(times))))))
		return map(lambda x: Trajectory(map(lambda t, pos: [t, pos], times, x)), trajs)
	@staticmethod
	def getCholCovar(params, times):
		covar = FractionalBrownianMotion1D.getCovar(params, times)
		chol_covar = sp.linalg.cholesky(covar, lower=True)
		return chol_covar
	@staticmethod
	def getCovar(params, times):
		n = len(times)
		sigma = params[0]
		H = params[1]
		two_H = 2 * H
		t_pow_h = map(lambda t: math.pow(t, two_H), times)
		half_sigma_sq = 0.5 * sigma * sigma
		covar = np.zeros([n, n])
		for i in range(0, n):
			covar[i][i] = half_sigma_sq * 2 * t_pow_h[i]
			for j in range(0, i):
				covar[i][j] = half_sigma_sq * (t_pow_h[i] + t_pow_h[j] - math.pow(times[i] - times[j], two_H))
				covar[j][i] = covar[i][j]
		return covar
	@staticmethod
	def test(params=[1., .75], numTrajs=100, plot=True):
		trajs = FractionalBrownianMotion1D.getTrajectory(params, np.arange(0, 10, .1), numTrajs=numTrajs)
		s = Sproid(trajectories=trajs)
		s.getResults(plot=plot)

#for lm, we have to import a C library to compute the log-likelihood
lmlib = np.ctypeslib.load_library('lm', '.')
lmLogLikelihood = lmlib.log_likelihood
lmLogLikelihood.restype = ctypes.c_double
lmLogLikelihood.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned'), ctypes.c_int]
class LevyMotion1D(StochasticProcess):
	def name(self):
		return "Levy motion 1D"
	def getNumParams(self):
		return 3
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
	def name(self):
		return "Symmetric Levy motion 1D"
	def getNumParams(self):
		return 2
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
	@staticmethod
	def getTrajectory(params, times, numTrajs=1):
		"""Generates a trajectory of a Symmetric Levy Motion in 1 dimension.

		times should all be non-negative and increasing
		"""
		if numTrajs > 1:
			return map(lambda x: SymmetricLevyMotion1D.getTrajectory(params, times, numTrajs=1)[0], range(0, numTrajs))
		else:
			previousT = 0
			alpha = params[0]
			beta = 0
			sigma = params[1]
			x = [levy.random(alpha, -beta) * sigma * math.pow(times[0], 1. / alpha)]
			for dt in diff(times):
				x.append(x[-1] + levy.random(alpha, -beta) * sigma * math.pow(dt, 1. / alpha))
			return [Trajectory(map(lambda t, pos: [t, pos], times, x))]
	@staticmethod
	def test(params=[1.5, 1.], numTrajs=100, plot=True):
		trajs = SymmetricLevyMotion1D.getTrajectory(params, np.arange(0, 10, .1), numTrajs=numTrajs)
		s = Sproid(trajectories=trajs)
		s.getResults(plot=plot)

class SymmetricLevyMotionWithDrift1D(SymmetricLevyMotion1D):
	def name(self):
		return "Symmetric Levy motion with drift 1D"
	def getNumParams(self):
		return 3
	def logPDF(self, params, trajectory):
		dxs = trajectory.getDisplacements1D(0)
		dts = trajectory.getDts()
		return symlmLogLikelihood(self.getAlpha(params), self.getVelocity(params), self.getLambda(params), np.array(dts, dtype=np.float64), np.array(dxs, dtype=np.float64), trajectory.len() - 1)
	def transformedPDF(self, params, trajectory):
		return self.logPDF(params, trajectory)
	def getVelocity(self, params):
		return self.transformParam(params, 2)

def power_law_clock(t, params, firstParamIndex):
	return math.pow(t, params[firstParamIndex])

bm = BrownianMotion1D([0.1], [10.0])
bmplc = StochasticProcessWithNonlinearClock(BrownianMotion1D, power_law_clock, [0.1, 0.25], [10.0, 1.0])
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

def summarizeResultsList(results, sps, plot=True):
	numTrajs = len(results)
	#collect the parameters, ics and winner counts for each stochastic process
	param_results = [[] for sp in sps]
	ic_results = [[] for sp in sps]
	winner_count = [0 for sp in sps]
	for i in range(0, numTrajs):
		result = results[i]
		winner_index = 0
		winner_ic = float("inf")
		for j in range(0, len(sps)):
			param_results[j].append(result[j][2])
			ic_results[j].append(result[j][0])
			if result[j][0] < winner_ic:
				winner_ic = result[j][0]
				winner_index = j
		winner_count[winner_index] += 1
	#print the average ICS
	print "Mean IC result:"
	for i in range(0, len(sps)):
		print str(sum(ic_results[i]) / numTrajs) + " " + sps[i].name()
	#determine the winner (the one with the best IC most often)
	winner_index = 0
	for i in range(1, len(sps)):
		if winner_count[i] > winner_count[winner_index]:
			winner_index = i
	print sps[winner_index].name() + " is the winner."
	for i in range(0, len(sps)):
		print str(winner_count[i]) + " wins for " + sps[i].name()
	#print the mean and variance for each of the parameters
	param_avgs = map(lambda x: map(lambda y: np.mean(y), zip(*x)), param_results)
	param_stds = map(lambda x: map(lambda y: np.std(y) , zip(*x)), param_results)
	for i in range(0, len(sps)):
		print "Average ML parameters for " + sps[i].name() + ":"
		for j in range(0, len(param_avgs[i])):
			print str(param_avgs[i][j]) + " +- " + str(param_stds[i][j])
	#make a scatter plot of the first two params if possible
	if len(param_results[winner_index][0]) > 1 and plot == True:
		plt.scatter(map(lambda i: param_results[winner_index][i][0], range(0, numTrajs)), map(lambda i: param_results[winner_index][i][1], range(0, numTrajs)))
		plt.show()

def summarizeResultsFolder(folder):
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
