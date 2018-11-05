# tests for ot functions
import utils
import numpy as np
from numpy.testing import assert_allclose
import otfunctions


def test_wkl():
    # test Kl Unbalanced OT divergence
    rng = np.random.RandomState(1729)
    n_features = 100
    epsilon = 1. / n_features  # entropy weight
    gamma = 5.  # KL constraint weight
    M = utils.groundmetric(n_features, normed=True)

    p, q = rng.rand(2, n_features)

    f, log = otfunctions.otkl(p=p, q=q, M=M, epsilon=epsilon, gamma=gamma,
                              tol=1e-6, returnmarginals=False,
                              returnlog=True, stable=False)

    # constraints
    assert_allclose(log["cstr"][-1], 0, atol=1e-3)


def test_wkl_log():
    # test Semi-Kl Unbalanced OT divergence
    rng = np.random.RandomState(1729)
    n_features = 100
    epsilon = 0.1 / n_features  # entropy weight
    gamma = 5.  # KL constraint weight
    M = utils.groundmetric(n_features, normed=True)

    p, q = rng.rand(2, n_features)
    f, log = otfunctions.otkl(p=p, q=q, M=M, epsilon=epsilon, gamma=gamma,
                              tol=1e-6, returnmarginals=False,
                              returnlog=True, stable=True)

    # constraints
    assert_allclose(log['cstr'][-1], 0, atol=1e-5, rtol=1e-5)


def test_wlog_match():
    # test KL Unblanced divergence log-domain
    rng = np.random.RandomState(1729)
    n_features = 100

    epsilon = 1. / n_features  # entropy weight
    gamma = 5.  # KL constraint weight

    M = utils.groundmetric(n_features, normed=True)

    p, q = rng.rand(2, n_features)

    out1 = otfunctions.otkl(p=p, q=q, M=M, epsilon=epsilon, gamma=gamma,
                            tol=1e-6, returnmarginals=False)

    out2 = otfunctions.otkl(p=p, q=q, M=M, epsilon=epsilon, gamma=gamma,
                            tol=1e-6, returnmarginals=False, stable=False)

    assert_allclose(out1, out2, atol=1e-5)


def test_2d_match():
    # test kernal separability in log-domain
    rng = np.random.RandomState(1729)

    width = 10
    n_features = width ** 2

    M = utils.groundmetric(width)
    Mbig = utils.groundmetric2d(n_features, normed=False)
    median = utils.median(Mbig)

    M /= median
    Mbig /= median

    epsilon = 1. / n_features
    stable = False

    gamma = 1.
    p = rng.rand(width, width)
    q = rng.rand(width, width)
    p[0, 0] = 0.8
    q[0, 1] = 1.

    f, log = otfunctions.otkl(p.flatten(), q.flatten(), M=Mbig,
                              epsilon=epsilon, gamma=gamma,
                              tol=1e-6, returnlog=True, stable=stable)

    f2d, log2d = otfunctions.otkl(p, q, M=M, epsilon=epsilon, gamma=gamma,
                                  tol=1e-6, returnlog=True, stable=stable)

    assert_allclose(log2d['cstr'][-1], 0, atol=1e-5, rtol=1e-5)

    assert_allclose(f, f2d, atol=1e-5)
