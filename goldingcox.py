from sproid import *

#s = Sproid(filename='goldingcox/goldingcox.yaml', downsamples=100)
s = Sproid(filename='goldingcox/goldingcox.yaml')
bm = BrownianMotion1D([1e-3], [1e0])
fbm = FractionalBrownianMotion1D([1e-3, 0.2], [1e0, 0.9])
slm = SymmetricLevyMotion1D([0.5, 1e-3], [1.999, 1e0])
bmplc = StochasticProcessWithNonlinearClock(BrownianMotion1D, power_law_clock, [1e-3, 0.25], [1e0, 2.0])

s.getResults(plot=False, sps=[bm, fbm, slm, bmplc], posIndex=0)
s.getResults(plot=False, sps=[bm, fbm, slm, bmplc], posIndex=1)
