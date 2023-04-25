import numpy as np
from matrix_chain import optimal_matrix_chain_cost
from scalene import scalene_profiler

rng = np.random.default_rng(seed=123)
tests = rng.integers(10, 1000, size=(10_000, 10))

scalene_profiler.start()
for test in tests:
    optimal_matrix_chain_cost(test)
scalene_profiler.stop()
