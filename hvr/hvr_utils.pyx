from libc.math cimport erf, exp, expm1, log, log1p, pi, sqrt

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

cpdef double norm_logl(double x, double m, double s):
    """Normal log-likelihood function."""
    return logsqrt2pi - 0.5*log(s) - 0.5*((x - m) / s)**2

cpdef double truncnorm_pdf(double x, double a, double b, double mu=0.5, double sigma=0.2):
    """Custom definition of the log of the truncated normal pdf."""
    cdef double p, z, alpha, beta, eta;
    beta = (b - mu) / sigma
    alpha = (a - mu) / sigma
    eta = (max(min(x,b),a) - mu) / sigma
    z = logdiffexp(psi(beta), psi(alpha))
    p = norm_pdf(eta) - log(sigma) - z
    return p

cpdef double emission_nvar(int c, double pi0, double alpha):
    """Emission distribution for number of variants."""
    pass

cpdef double emission_callrate(double[:] call_rates):
    """Emission distribution for accounting for variation in call-rate."""
    pass


def forward_algo(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25):
    """Helper function for forward algorithm loop-optimization."""
    cdef int i,j,n,m;
    cdef float di;
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0,K1 = create_index_arrays(karyotypes)
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = log(1.0 / m)
    for j in range(m):
        m_ij = mat_dosage(mat_haps[:, 0], states[j])
        p_ij = pat_dosage(pat_haps[:, 0], states[j])
        # This is in log-space ...
        cur_emission = emission_baf(
                bafs[0],
                m_ij,
                p_ij,
                pi0=pi0,
                std_dev=std_dev,
                k=ks[j],
            )
        alphas[j,0] += cur_emission
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        # This should get the distance dependent transition models ...
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            # This is in log-space ...
            cur_emission = emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
            alphas[j, i] = cur_emission + logsumexp(A_hat[:, j] + alphas[:, (i - 1)])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)

def backward_algo(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i,j,n,m;
    cdef float di;
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0,K1 = create_index_arrays(karyotypes)
    betas = np.zeros(shape=(m, n))
    betas[:,-1] = log(1)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        # The matrices are element-wise multiplied so add in log-space ...
        di = pos[i+1] - pos[i]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        # Calculate the full set of emissions
        cur_emissions = np.zeros(m)
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i+1], states[j])
            p_ij = pat_dosage(pat_haps[:, i+1], states[j])
            # This is in log-space as well ...
            cur_emissions[j] = emission_baf(
                    bafs[i + 1],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
        for j in range(m):
            # This should be the correct version here ...
            betas[j,i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            for j in range(m):
                m_ij = mat_dosage(mat_haps[:, i], states[j])
                p_ij = pat_dosage(pat_haps[:, i], states[j])
                # This is in log-space as well ...
                cur_emission = emission_baf(
                        bafs[i],
                        m_ij,
                        p_ij,
                        pi0=pi0,
                        std_dev=std_dev,
                        k=ks[j],
                    )
                # Add in the initialization + first emission?
                betas[j,i] += log(1/m) + cur_emission
        # Do the rescaling here ...
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)

def viterbi_algo(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25):
    """Cython implementation of the Viterbi algorithm for MLE path estimation through states."""
    cdef int i,j,n,m;
    cdef float di;
    n = bafs.size
    m = len(states)
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1.0 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0,K1 = create_index_arrays(karyotypes)
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            deltas[j,i] = np.max(deltas[:,i-1] + A_hat[:,j])
            deltas[j,i] += emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
            psi[j, i] = np.argmax(deltas[:, i - 1] + A_hat[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi
