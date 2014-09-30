1. Installing
After the software has been downloaded and extracted, it is necessary to build shared libraries.
This is achieved by running
	make

2. Testing
To test the setup, run
	python runTests.py

3. Using
To use SPROID, trajectory data should be entered in a YAML format (see http://yaml.org).
The format used by SPROID is (see test.yaml)
	Trajectories:
	- [ [0, 1], [.1, 2], [.5, 1] ]
	- [ [0, 1, 2], [1, 2, 3], [2, 1, 4], [3, 2, 5], [4, 1, 6] ]
	- [ [0.5, 3, 6], [1.5, 1, 0], [2.5, -1, -4], [3.5, 3, 7] ]
The file consists of a list of trajectories.
Each trajectory is represented as a list of points in space-time.
The point may be in a space with arbitrary dimension, but the time component must be listed first (that is, [t, x, y, ...]).
After creating a file with the trajectory data, SPROID can be utilized as follows
	#import sproid code
	from sproid import *

	#load trajectory data
	s = Sproid(filename='filename.yaml')

	#Now we create a number of stochastic processes with unknown parameters
	#At present SPROID only supports 1D processes
	#Brownian motion with diffusion coefficient between 1e-3 and 1
	bm = BrownianMotion1D([1e-3], [1e0])
	#fractional Brownian with diffusion coefficient between 1e-3 and 1; and Hurst exponent between 0.2 and 0.9
	fbm = FractionalBrownianMotion1D([1e-3, 0.2], [1e0, 0.9])
	#symmmemtric Levy motion with alpha between 0.5 and 1.999 and diffusion coefficient between 1e-3 and 1
	slm = SymmetricLevyMotion1D([0.5, 1e-3], [1.999, 1e0])
	#Brownian motion with a power-law clock with diffusion coefficient between 1e-3 and 1; and power-law (for the clock) between 0.25 and 2.0
	bmplc = StochasticProcessWithNonlinearClock(BrownianMotion1D, power_law_clock, [1e-3, 0.25], [1e0, 2.0])

	#analyze the trajectories using the stochastic processes
	#analyze the first spatial coordinate of the trajectories
	s.getResults(plot=False, sps=[bm, fbm, slm, bmplc], posIndex=0)
	#analyze the second spatial coordinate of the trajectories
	s.getResults(plot=False, sps=[bm, fbm, slm, bmplc], posIndex=1)
Different sets of stochastic processes and parameters may be used.
Note that getResults(...) runs in parallel by default using 4 processes.
