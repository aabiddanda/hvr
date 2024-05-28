from libc.math cimport erf, exp, expm1, lgamma, log, log1p, pi, sqrt

import numpy as np


cdef double sqrt2 = sqrt(2.);
cdef double sqrt2pi = sqrt(2*pi);
cdef double logsqrt2pi = log(1/sqrt2pi)

cpdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick"""
    cdef int i,n;
    cdef double m = -1e32;
    cdef double c = 0.0;
    n = x.size
    for i in range(n):
        m = max(m,x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double logdiffexp(double a, double b):
    """Log-sum-exp trick but for differences."""
    return log(exp(a) - exp(b) + 1e-124)

cpdef double logaddexp(double a, double b):
    cdef double m = -1e32;
    cdef double c = 0.0;
    m = max(a,b)
    c = exp(a - m) + exp(b - m)
    return m + log(c)

cpdef double logmeanexp(double a, double b):
    """Apply a logmeanexp routine for two numbers."""
    cdef double m = -1e32;
    cdef double c = 0.0;
    m = max(a,b)
    c = exp(a - m) + exp(b - m)
    return m + log(c) - 2.0

cdef double log1mexp(double a):
    """Log of 1 - e^-x."""
    if a < 0.693:
        return log(-expm1(-a))
    else:
        return log1p(-exp(-a))

cdef double psi(double x):
    """CDF for a normal distribution function in log-space."""
    if x < -4:
        return logsqrt2pi - 0.5*(x**2) - log(-x)
    else:
        return log((1.0 + erf(x / sqrt2))) - log(2.0)

cdef double norm_pdf(double x):
    """PDF for the normal distribution function in log-space.

    NOTE: at some point we might want to generalize this to include mean-shift and stddev.
    """
    return logsqrt2pi - 0.5*(x**2)

cdef double beta_pdf(double x, double a, double b):
    """PDF for the beta distribution function in log-space."""
    return (a - 1) * log(x) + (b - 1)*log(1-x) + lgamma(a+b) - lgamma(a) - lgamma(b)

cpdef double norm_logl(double x, double m, double s):
    """Normal log-likelihood function."""
    return logsqrt2pi - 0.5*log(s) - 0.5*((x - m) / s)**2

cpdef double emission_nvar(int c, double lambda0=1.0, double alpha = 1.0):
    """Emission distribution for number of variants."""
    return -alpha*lambda0 + c * log(alpha*lambda0)

cpdef double emission_callrate(double call_rate = 1.0, double a=1.0, double b=1.0):
    """Emission distribution for the mean call-rate in a window."""
    if (call_rate == 0.0) or (call_rate == 1.0):
        return 0.0
    else:
        return beta_pdf(call_rate, a, b)

def forward_algo(int[:] cnts, double[:] call_rates, double[:] pos, double pi0=0.2, double eps=1e-3, double lambda0=1.0, double alpha=2.0, double a0=1.0, double b0=1.0, double a1=0.5, double b1=0.5):
    """Helper function for forward algorithm loop-optimization."""
    cdef int i,j,n,m;
    cdef float di;
    assert cnts.size == call_rates.size
    assert cnts.size == pos.size
    n = cnts.size
    m = 2
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = log(1.0 / m)
    alphas[0,0] += emission_nvar(cnts[0], lambda0=lambda0, alpha=1.0)
    alphas[0,0] += emission_callrate(call_rates[0], a=a0, b=b0)
    alphas[1,0] += emission_nvar(cnts[0], lambda0=lambda0, alpha=alpha)
    alphas[1,0] += emission_callrate(call_rates[0], a=a1, b=b1)
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = np.array([[-di, log1mexp(di)],[log1mexp(di), -di]])
        cur_emission0 = emission_nvar(cnts[i], lambda0=lambda0, alpha=1.0) + emission_callrate(call_rates[i], a=a0, b=b0)
        cur_emission1 = emission_nvar(cnts[i], lambda0=lambda0, alpha=alpha) + emission_callrate(call_rates[i],  a=a1, b=b1)
        alphas[0, i] = cur_emission0 + logsumexp(A_hat[:, 0] + alphas[:, (i - 1)])
        alphas[1, i] = cur_emission1 + logsumexp(A_hat[:, 1] + alphas[:, (i - 1)])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, sum(scaler)

def backward_algo(int[:] cnts, double[:] call_rates, double[:] pos, double lambda0=1.0, double alpha=2.0, double a0=1.0, double b0=1.0, double a1=0.5, double b1=0.5):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i,j,n,m;
    cdef float di;
    assert cnts.size == call_rates.size
    assert cnts.size == pos.size
    n = cnts.size
    m = 2
    betas = np.zeros(shape=(m, n))
    betas[:,-1] = log(1)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        # The matrices are element-wise multiplied so add in log-space ...
        di = pos[i+1] - pos[i]
        A_hat = np.array([[-di, log1mexp(di)],[log1mexp(di), -di]])
        # Calculate the full set of emissions
        cur_emissions = np.zeros(m)
        cur_emissions[0] = emission_nvar(cnts[i+1], lambda0=lambda0, alpha=1.0) + emission_callrate(call_rates[i+1], a=a0, b=b0)
        cur_emissions[1] = emission_nvar(cnts[i+1], lambda0=lambda0, alpha=alpha) + emission_callrate(call_rates[i+1], a=a1, b=b1)

        # This should be the correct version here ...
        betas[0,i] = logsumexp(A_hat[:, 0] + cur_emissions + betas[:, (i + 1)])
        betas[1,i] = logsumexp(A_hat[:, 1] + cur_emissions + betas[:, (i + 1)])

        if i == 0:
            cur_emissions = np.zeros(m)
            cur_emissions[0] = emission_nvar(cnts[i], lambda0=lambda0, alpha=1.0) + emission_callrate(call_rates[i], a=a0, b=b0)
            cur_emissions[1] = emission_nvar(cnts[i], lambda0=lambda0, alpha=alpha) + emission_callrate(call_rates[i], a=a1, b=b1)

            # Add in the initialization + first emission?
            betas[0,i] += log(1/m) + cur_emissions[0]
            betas[1,i] += log(1/m) + cur_emissions[1]
        # Do the rescaling here ...
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, sum(scaler)

def viterbi_algo(int[:] cnts, double[:] call_rates, double[:] pos, double lambda0=1.0, double alpha=2.0, double a0=1.0, double b0=1.0, double a1=0.5, double b1=0.5):
    """Cython implementation of the Viterbi algorithm for MLE path estimation through states."""
    cdef int i,j,n,m;
    cdef float di;
    assert cnts.size == call_rates.size
    assert cnts.size == pos.size
    n = cnts.size
    m = 2
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1.0 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = [[-di, log1mexp(di)],[log1mexp(di), -di]]
        for j in range(m):
            deltas[0,i] = np.max(deltas[:,i-1] + A_hat[:,0])
            deltas[0,i] += emission_nvar(cnts[i], lambda0=lambda0, alpha=1.0) + emission_callrate(call_rates[i], a=a0, b=b0)
            psi[0, i] = np.argmax(deltas[:, i - 1] + A_hat[:, 0]).astype(int)
            deltas[1,i] = np.max(deltas[:,i-1] + A_hat[:,1])
            deltas[1,i] += emission_nvar(cnts[i], lambda0=lambda0, alpha=alpha) + emission_callrate(call_rates[i], a=a1, b=b1)
            psi[1, i] = np.argmax(deltas[:, i - 1] + A_hat[:, 1]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, deltas, psi
