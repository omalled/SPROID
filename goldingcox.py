import * from sproid

s = Sproid(filename='goldingcox.yaml', downsamples=100)
bm = BrownianMotion1D([1e-4], [1e0], logParams=[True])
fbm = FractionalBrownianMotion1D([1e-4, 0.2], [1e0, 0.9], logParams=[True, False])
slm = SymmetricLevyMotion1D([0.5, 1e-4], [1.999, 1e0], logParams=[False, True])
bmplc = StochasticProcessWithNonlinearClock(BrownianMotion1D, power_law_clock, [1e-4, 0.25], [1e0, 2.0])

s.getResults(plot=False, sps=[bm, fbm, slm, bmplc])
