"""Unbalanced Optimal transport functions KL div + convultions.
"""
import numpy as np
import warnings
from utils import check_zeros
from sys import float_info
import utils
try:
    import cupy as cp
    get_module = cp.get_array_module
except:
    get_module = lambda x: np


def wkl_(p, q, M, K=None, epsilon=0.01, gamma=1., maxiter=20000, tol=1e-5,
        returnlog=False, returnmarginals=False):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_features, n_hists)
        Must be non-negative.
    q: numpy array (n_features, )
        Must be non-negative.
    M: numpy array (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    gamma: float > 0.
        Kullback-Leibler marginal constraint weight w.r.t q.
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    xp = get_module(M)
    n_features = M.shape[0]
    if K is None:
        K = xp.exp(- M / epsilon)
    Kb = K.mean(axis=1)

    if p.ndim > 1:
        p = p.reshape(len(p), -1).copy()
        q = q.reshape(len(q), -1).copy()
        n_hists = p.shape[-1]
        Kb = Kb.reshape(-1, 1)
    frac = gamma / (gamma + epsilon)

    a, b = xp.ones((2, n_features))
    log = {'cstr': [], 'obj': [], 'flag':0, 'objexact':[], 'b':[], 'a':[]}
    f0 = K.sum()
    f = f0
    cstr = 10
    for i in range(maxiter):
        a = (p / Kb) ** frac
        b = (q / K.T.dot(a)) ** frac
        oldf = f
        Kb = K.dot(b)
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma)
        cstr = abs(f - oldf) / max(abs(f), abs(oldf), 1)
        log["cstr"].append(cstr)
        log["obj"].append(f)
        if cstr < tol:
            break

    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    if not log['obj']:
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma)
    out = f

    if returnmarginals:
        marginals = a * Kb, Ka * b
        out = out, marginals

    if returnlog:
        return out, log

    return out


def wkl_log(p, q, M, K=None, epsilon=0.01, gamma=1., maxiter=20000, tol=1e-5,
            returnlog=False, returnmarginals=False):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_features, n_hists)
        Must be non-negative.
    q: numpy array (n_features, )
        Must be non-negative.
    M: numpy array (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    gamma: float > 0.
        Kullback-Leibler marginal constraint weight w.r.t q.
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginals: boolean.
        default False. if True, returns the scaling vectors.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    xp = get_module(M)
    n_features = M.shape[0]
    p, q, M = p, q, M
    out = xp.zeros((2, n_features))

    if K is None:
        K = xp.exp(- M / epsilon)
    Ks = K.copy()
    Kb = K.sum(axis=1)

    if p.ndim > 1:
        p = p.reshape(len(p), -1).copy()
        q = q.reshape(len(q), -1).copy()
        n_hists = p.shape[-1]
        Kb = Kb.reshape(-1, 1)

    frac = gamma / (gamma + epsilon)

    a, b = xp.ones((2, n_features))
    u, v = xp.zeros((2, n_features))
    log = {'cstr': [], 'obj': [], 'flag':0, 'objexact':[]}
    f0 = K.sum()
    f = f0
    cstr = 10
    for i in range(maxiter):
        aold = a.copy()
        a = (p / (Kb + 1e-16)) ** frac * xp.exp(- u / (epsilon + gamma))
        Ka = Ks.T.dot(a)
        b = (q / (Ka + 1e-16)) ** frac * xp.exp(- v / (epsilon + gamma))

        if (a > 1e5).any() or (b > 1e5).any():
            u += epsilon * xp.log(a + 1e-16)
            v += epsilon * xp.log(b + 1e-16)
            Ks = xp.exp((u.reshape(-1, 1) + v.reshape(1, -1) - M) / epsilon)
            b = xp.ones(n_features)
        Kb = Ks.dot(b)

        oldf = f
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma, u=u)

        cstr = abs(f - oldf) / max(abs(f), abs(oldf), 1)
        log["cstr"].append(cstr)
        log["obj"].append(f)

        if cstr < tol:
            break
    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    if not len(['obj']):
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma, u=u)
    out = f

    if returnmarginals:
        marginals = a * Kb, Ka * b
        out = out, marginals

    if returnlog:
        return out, log

    return out


def wklimg_log(p, q, M, epsilon=0.01, gamma=1., maxiter=20000, tol=1e-5,
               returnlog=False, returnmarginals=False):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    q: numpy array (width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    gamma: float > 0.
        Kullback-Leibler marginal constraint weight w.r.t q.
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    width = M.shape[0]
    xp = get_module(M)
    frac = gamma / (gamma + epsilon)

    aold, bold = xp.zeros_like(p), xp.zeros_like(p)
    b = bold.copy()
    Kb = utils.kls1d(b, - M / epsilon)
    log = {'cstr': [], 'obj': [], 'flag':0, 'a':[], 'b':[]}
    f0 = xp.exp(- M / epsilon).sum() ** 2
    fold = f0
    cstr = 10
    logp, logq = xp.log(p), xp.log(q)
    for i in range(maxiter):
        a = frac * (logp - Kb)
        Ka = utils.kls1d(a, - M.T / epsilon)
        b = frac * (logq - Ka)
        Kb = utils.kls1d(b, - M / epsilon)

        f = utils.wklobjective_log(a, Kb, p, q, f0, epsilon, gamma)
        cstr = abs(f - fold) / max(abs(f), abs(fold), 1)
        fold = f

        log["cstr"].append(cstr)

        log["obj"].append(f)
        if cstr < tol:
            break

    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    if not log['obj']:
        f = utils.wklobjective_log(a, Kb, p, q, f0, epsilon, gamma)
    out = f

    if returnmarginals:
        marginals = xp.exp(a + Kb), xp.exp(Ka + b)
        out = out, marginals

    if returnlog:
        return out, log

    return out


def wklimg(p, q, M, epsilon=0.01, gamma=1., maxiter=20000, tol=1e-5,
           returnlog=False, returnmarginals=False):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    q: numpy array (width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    gamma: float > 0.
        Kullback-Leibler marginal constraint weight w.r.t q.
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    width = M.shape[0]
    xp = get_module(M)
    frac = gamma / (gamma + epsilon)
    K = xp.exp(- M / epsilon)
    aold, bold = xp.ones_like(p), xp.ones_like(p)
    b = bold.copy()
    Kb = utils.klconv1d(b, K)
    log = {'cstr': [], 'obj': [], 'flag':0, 'a':[], 'b':[]}
    f0 = K.sum() ** 2
    fold = f0
    cstr = 10
    for i in range(maxiter):
        a = (p / Kb) ** frac
        Ka = utils.klconv1d(a, K.T)
        b = (q / Ka) ** frac
        Kb = utils.klconv1d(b, K)

        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma)
        cstr = abs(f - fold) / max(abs(f), abs(fold), 1)
        fold = f

        log["cstr"].append(cstr)

        log["obj"].append(f)
        if cstr < tol:
            break

    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    if not log['obj']:
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma)
    out = f

    if returnmarginals:
        marginals = a * Kb, Ka * b
        out = out, marginals

    if returnlog:
        return out, log

    return out


def otkl(p, q, M, stable=True, **kwargs):
    """OT KL general function."""

    if q.ndim > 1:
        if stable:
            return wklimg_log(p + 1e-10, q + 1e-10, M, **kwargs)
        return wklimg(p + 1e-10, q + 1e-10, M, **kwargs)

    if stable:
        return wkl_log(p + 1e-10, q + 1e-10, M, **kwargs)
    return wkl_(p + 1e-10, q + 1e-10, M, **kwargs)
