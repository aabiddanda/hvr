"""Test suite for HVR calculations."""
import numpy as np
from hvr_utils import backward_algo, emission_callrate, emission_nvar, forward_algo
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import beta, poisson


@given(
    c=st.integers(min_value=0, max_value=1),
    lamb=st.floats(min_value=0.0, max_value=20, exclude_min=True),
)
def test_emission_nvar(c, lamb):
    """Test that the emission for the call rate is correct."""
    true_emiss = emission_nvar(c, lambda0=lamb, alpha=1.0)
    pois_emiss = poisson.logpmf(c, mu=lamb)
    assert np.isclose(true_emiss, pois_emiss, atol=1e-06)


@given(
    r=st.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True),
    a=st.floats(min_value=1e-5, max_value=20, exclude_min=True),
    b=st.floats(min_value=1e-5, max_value=20, exclude_min=True),
)
def test_emission_callrate(r, a, b):
    """Test that the emission for call rate is correct."""
    true_emiss = emission_callrate(r, a=a, b=b)
    beta_emiss = beta.logpdf(r, a, b)
    assert np.isclose(true_emiss, beta_emiss, atol=1e-06)


@given(r=st.integers(min_value=5, max_value=100))
def test_forward_algo(r):
    """Test that the forward log-likelihood makes sense."""
    np.random.seed(r)
    cnts = poisson.rvs(mu=1.0, size=r)
    callrates = beta.rvs(a=1.0, b=1.0, size=r)
    pos = np.linspace(0, 1e6, r)
    _, _, loglik = forward_algo(cnts, callrates, pos)
    assert loglik < 0


@given(r=st.integers(min_value=5, max_value=100))
def test_backward_algo(r):
    """Test that the backward log-likelihood makes sense."""
    np.random.seed(r)
    cnts = poisson.rvs(mu=1.0, size=r)
    callrates = beta.rvs(a=1.0, b=1.0, size=r)
    pos = np.linspace(0, 1e6, r)
    _, _, loglik = backward_algo(cnts, callrates, pos)
    assert loglik < 0


@given(r=st.integers(min_value=5, max_value=100))
def test_loglik_similarity(r):
    """Test that the log-likelihood makes sense."""
    np.random.seed(r)
    cnts = poisson.rvs(mu=1.0, size=r)
    callrates = beta.rvs(a=1.0, b=1.0, size=r)
    pos = np.linspace(0, 1e6, r)
    _, _, fwd_loglik = forward_algo(cnts, callrates, pos)
    _, _, bwd_loglik = backward_algo(cnts, callrates, pos)
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-5)
