"""Testing for the optimal matrix chain ordering."""

import numpy as np
from matrix_chain import multi_dot, optimal_matrix_chain_cost
import pytest

class DotCostTracker:
    def __init__(self):
        self.cost = 0

    def dot(self, a, b):
        a1, a2 = a.shape
        b1, b2 = b.shape
        assert a2 == b1
        self.cost += a1 * a2 * b2
        return np.dot(a, b)


def dp_matrix_cost(p):
    """
    Return a np.array that encodes the optimal order of mutiplications.

    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.

    Also return the cost matrix if `return_costs` is `True`

    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.

        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])

    """
    n = len(p) - 1
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = np.zeros((n, n), dtype=object)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = np.empty((n, n), dtype=object)

    for el in range(1, n):
        for i in range(n - el):
            j = i + el
            m[i, j] = np.Inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return m[0, -1]


def test_single_case():
    rng = np.random.default_rng(seed=1234)
    dims = rng.choice(np.arange(500, 1500), size=500, replace=False)
    answer = 241357185845

    assert optimal_matrix_chain_cost(dims)[1] == answer


@pytest.mark.parametrize(
    "seed,n_test,n_dim,lo,hi",
    [(10, 100, 8, 100, 150), (11, 100, 16, 10, 30), (12, 100, 32, 10, 30)],
)
def test_random(seed, n_test, n_dim, lo, hi):
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    for _i in range(n_test):
        dims = rng.integers(lo, hi, size=n_dim)
        # assert optimal_matrix_chain_cost(dims)[1] == dp_matrix_cost(dims)
        mats = [rng.standard_normal((dims[i], dims[i+1])) for i in range(len(dims) - 1)]
        tracer = DotCostTracker()
        assert np.allclose(multi_dot(mats, tracer.dot), np.linalg.multi_dot(mats))
        assert tracer.cost == dp_matrix_cost(dims)


@pytest.mark.slow
def test_random_200():
    rng = np.random.default_rng(seed=1234)
    for _i in range(1000):
        dims = rng.uniform(0.1, 10, size=200).astype(np.double)
        assert optimal_matrix_chain_cost(dims)[1] == pytest.approx(dp_matrix_cost(dims))
