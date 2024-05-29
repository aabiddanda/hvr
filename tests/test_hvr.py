"""Test suite for HVR calculations."""
import numpy as np
from hvr_utils import backward_algo, emission_callrate, emission_nvar, forward_algo
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import beta, poisson

from hvr import HVR


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
    """Test that the log-likelihood is equal between forward and backward settings."""
    np.random.seed(r)
    cnts = poisson.rvs(mu=1.0, size=r)
    callrates = beta.rvs(a=1.0, b=1.0, size=r)
    pos = np.linspace(0, 1e6, r)
    _, _, fwd_loglik = forward_algo(cnts, callrates, pos)
    _, _, bwd_loglik = backward_algo(cnts, callrates, pos)
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-5)


@given(r=st.integers(min_value=5, max_value=100))
def test_forward_backward(r):
    """Test that the forward-backward algorithm is running properly."""
    np.random.seed(r)
    cnts = poisson.rvs(mu=1.0, size=r)
    callrates = beta.rvs(a=1.0, b=1.0, size=r)
    pos = np.linspace(0, 1e6, r)
    cur_hvr = HVR("test.vcf.gz")
    # Creating the HVR object ...
    cur_hvr.chrom_cnts = {"I": cnts}
    cur_hvr.chrom_call_rate = {"I": callrates}
    cur_hvr.chrom_pos = {"I": pos}
    cur_hvr.est_lambda0()
    cur_hvr.est_beta_params()
    gamma_dict = cur_hvr.forward_backward(
        lambda0=cur_hvr.lambda0, alpha=2.0, a0=cur_hvr.a0, b0=cur_hvr.b0, a1=3, b1=3
    )
    gammas = gamma_dict["I"]
    gammas_raw = np.exp(gammas)
    assert np.all(np.isclose(np.sum(gammas_raw, axis=0), 1.0, atol=1e-4))
