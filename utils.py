"""Utils for OT."""
import numpy as np
try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    get_module = lambda x: np


def check_zeros(P, q, M, threshold=0):
    """Remove zero-entries.

    Parameters
    ----------
    P: array-like (n_features, n_hists)
        postive histograms stacked as columns.
    q: array-like (n_features,)
        positive reference histogram.
    M: array-like (n_features, n_features)
        cost matrix.

    """
    left = P.reshape(len(P), -1).any(axis=-1) > threshold
    right = q > threshold
    M2 = M[left, :][:, right]
    P2 = P[left]
    q2 = q[right]
    return P2, q2, M2


def median(x):
    """Compute median."""
    xp = get_module(x)
    x = x.flatten()
    n = len(x)
    s = xp.sort(x)
    m_odd = xp.take(s, n // 2)
    if n % 2 == 1:
        return m_odd
    else:
        m_even = xp.take(s, n // 2 - 1)
        return (m_odd + m_even) / 2


def format_sc(n):
    """Return a LaTeX scientifc format for a float n.

    Parameters
    ----------
    n: float.

    Returns
    -------
    str.
        scientific writing of n in LaTeX.

    """
    a = "%.2E" % n
    b = a.split('E')[0] + ' '
    p = str(int(a.split('E')[-1]))
    if p == '0':
        return b
    return b + r'$10^{' + p + r'}$'


def groundmetric(n_features, p=2, normed=False):
    """Compute ground metric matrix on the 2D grid 0:`n_features` ^ 2.

    Parameters
    ----------
    n_features: int > 0.
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (n_features, n_features).

    """
    x = np.arange(0, n_features).reshape(-1, 1).astype(float)
    xx, yy = np.meshgrid(x, x)
    M = abs(xx - yy) ** p
    if normed:
        M /= median(M)
    return M


def groundmetric2d(n_features, p=2, normed=False):
    """Compute ground metric matrix on the 2D grid 0:`n_features` ^ 2.

    Parameters
    ----------
    n_features: int > 0.
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (n_features, n_features).

    """
    d = int(n_features ** 0.5)
    n_features = d ** 2
    M = groundmetric(d, p=2, normed=False)
    M = M[:, np.newaxis, :, np.newaxis] + M[np.newaxis, :, np.newaxis, :]
    M = M.reshape(n_features, n_features) ** (p / 2)
    if normed:
        M /= median(M)
    return M


def kl(p, q):
    """Compute Kullback-Leibler divergence.

    Compute the element-wise sum of:
    `p * log(p / q) + p - q`.

    Parameters
    ----------
    p: array-like.
        must be positive.
    q: array-like.
        must be positive, same shape and dimension of `p`.

    """
    xp = get_module(p)
    logpq = xp.log((p + 1e-16) / (q + 1e-16))
    kl = (p * logpq + q - p).sum()

    return kl


def wklobjective0(plan, p, q, K, epsilon, gamma):
    """Compute unbalanced ot objective function naively."""
    f = epsilon * kl(plan, K)
    margs = kl(plan.sum(axis=1), p) + kl(plan.sum(axis=0), q)
    f += gamma * margs

    return f


def wklobjective_log(a, Kb, p, q, Ksum, epsilon, gamma):
    """Compute unbalanced ot objective function for solver monitoring."""
    xp = get_module(a)
    n_hists = 1
    if a.ndim > 2:
        n_hists = a.shape[-1]

    aKb = xp.exp(a + Kb)
    f = gamma * kl(aKb, p)
    f += (aKb * (epsilon * a - epsilon - gamma)).sum()
    f += n_hists * epsilon * Ksum
    f += n_hists * gamma * q.sum()

    return f


def wklobjective(a, Kb, p, q, Ksum, epsilon, gamma, u=0):
    """Compute unbalanced ot objective function for solver monitoring."""
    xp = get_module(a)
    n_hists = 1
    if a.ndim > 2 or (a.ndim == 2 and a.shape[0] != a.shape[1]):
        n_hists = a.shape[-1]
    aKb = a * Kb
    f = gamma * kl(aKb, p)
    f += (aKb * (epsilon * xp.log(a + 1e-16) + u - epsilon - gamma)).sum()
    f += n_hists * epsilon * Ksum
    f += n_hists * gamma * q.sum()

    return f


def metricmedian(n_features):
    """Compute median of euclidean cost matrix.

    Parameters
    ----------
    n_features: int.
        size of histograms.

    Returns
    -------
    median: float.
        median of pairwise distance matrix M.

    """
    m = - np.sqrt(0.5 * n_features ** 2 + n_features - 0.25)
    m += 0.5 + n_features
    m = int(m)
    median = m ** 2

    return median


def compute_gamma(tau, M, p=None, q=None):
    """Compute the sufficient KL weight for a full mass minimum."""
    xp = get_module(M)

    if p is None or q is None:
        return max(0., - M.max() / (2 * xp.log(tau)))

    geomass = p.sum() * q.sum()
    gamma = p.dot(M.dot(q)) / geomass
    gamma /= 1 - tau ** 2

    return max(0., gamma)


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements."""
    xp = get_module(a)

    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -float('inf')

    a_max = xp.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~xp.isfinite(a_max)] = 0
    elif not xp.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = xp.asarray(b)
        tmp = b * xp.exp(a - a_max)
    else:
        tmp = xp.exp(a - a_max)

    # suppress warnings about log of zero
    s = xp.sum(tmp, axis=axis, keepdims=keepdims)
    if return_sign:
        sgn = xp.sign(s)
        s *= sgn  # /= makes more sense but we need zero -> zero
    out = xp.log(s)

    if not keepdims:
        a_max = xp.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def kls1d(img, C):
    """Compute log separable kernal application."""
    xp = get_module(C)
    x = logsumexp(C[xp.newaxis, :, :] + img[:, xp.newaxis, :], axis=-1)
    x = logsumexp(C.T[:, :, xp.newaxis] + x[:, xp.newaxis, :], axis=0)
    return x


# for lists, vectorized:
def kls(img, C):
    """Compute log separable kernal application."""
    xp = get_module(C)
    x = logsumexp(C[xp.newaxis, :, :, xp.newaxis] + img[:, xp.newaxis],
                  axis=-2)
    x = logsumexp(C.T[:, :, xp.newaxis, xp.newaxis] + x[:, xp.newaxis], axis=0)
    return x


def klconv1d(img, K):
    """Compute separable kernel application with pseudo convolutions."""
    X = K.dot(K.dot(img).T).T
    return X


def klconv(img, K):
    """Compute separable kernel application with pseudo convolutions."""
    xp = get_module(K)
    X = xp.vstack([K.dot(K.dot(im).T).T[xp.newaxis] for im in img.T])
    return X.T
