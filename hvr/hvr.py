"""Class for defining HMM for identifying highly variable regions in nematode genomes."""

import warnings

import numpy as np
from cyvcf2 import VCF
from hvr_utils import backward_algo, forward_algo, viterbi_algo
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import logsumexp as logsumexp_sp
from scipy.stats import *
from tqdm import tqdm


class HVR:
    """Class for HMM-based calling of hyper-variable regions in genomes."""

    def __init__(self, vcf_file):
        """Initialize the HVR class."""
        self.vcf_file = vcf_file
        self.chrom_pos = None
        self.chrom_cnts = None
        self.chrom_call_rate = None

    def generate_window_data(self, window_size=100, **kwargs):
        """Create the emission / position vectors at intervals of 100 basepairs."""
        assert window_size > 0
        if window_size <= 100:
            warnings.warn("Warning: window-size for estimation is <= 100 bp!")
        vcf = VCF(self.vcf_file, **kwargs)
        chroms = vcf.seqnames
        chrom_cnt_dict = {}
        chrom_call_rate_dict = {}
        chrom_pos_dict = {}
        for c in chroms:
            pos = []
            call_rate = []
            for v in tqdm(vcf(c)):
                pos.append(v.POS)
                call_rate.append(v.call_rate)
            if len(pos) > 0:
                # Create a binned representation of this ...
                bins = np.arange(np.min(pos), np.max(pos), window_size)
                res = binned_statistic(pos, np.ones(len(pos)), statistic="count", bins=bins)
                # Set the counts here ...
                cnts = res.statistic
                # NOTE: this is the mean-call-rate in that window ...
                res = binned_statistic(pos, call_rate, statistic=np.nanmean, bins=bins)
                call_rates = res.statistic
                pos = (bins[:-1] + bins[1:]) / 2
                assert pos.size == call_rates.size
                assert pos.size == cnts.size
                chrom_pos_dict[c] = pos
                chrom_cnt_dict[c] = cnts
                chrom_call_rate_dict[c] = call_rates
        # Actually set dictionaries as the object here ...
        self.chrom_cnts = chrom_cnt_dict
        self.chrom_pos = chrom_pos_dict
        self.chrom_call_rate = chrom_call_rate_dict

    def interpolate_rec_dist(self, recmap=None, rec_rate=1e-4):
        """Interpolate the recombination distance."""
        assert self.chrom_pos is not None
        assert (rec_rate > 0) and (rec_rate < 1)
        if recmap is None:
            for k in self.chrom_pos:
                cur_pos = self.chrom_pos[k]
                self.chrom_pos[k] = [(s + e / 2) * rec_rate for (s, e) in cur_pos]
        else:
            raise NotImplementedError(
                "Interpolation via a recombination map is not currently supported!"
            )

    def est_beta_params(self):
        """Estimate the null beta parameters across all the call-rates.

        NOTE: this is just using moment-matching for the beta distribution ...
        """
        xs = np.hstack([self.chrom_call_rate[i] for i in self.chrom_call_rate])
        mu = np.nanmean(xs)
        sigma2 = np.nanvar(xs)
        a = -(sigma2 + mu^2 - mu) / sigma2
        b =
        pass

    def est_lambda0(self):
        """Estimate the rate of variants."""
        pass

    def optimize_params(self, lambda0=1.0, algo="L-BFGS-B"):
        """Optimize the parameters."""
        assert self.lambda0 is not None
        assert algo in ["L-BFGS-B", "Powell"]
        opt_res = minimize(
            lambda x: -self.loglik(
                pi0=x[0],
                eps=1e-3,
                lambda0=self.lambda0,
                alpha=x[1],
                a0=x[2],
                b0=x[3],
                a1=x[4],
                b1=x[5],
            ),
            x0=[],
            method=algo,
            bounds=[
                (1e-3, 0.8),
                (1.0, 100.0),
                (1e-2, 1e2),
                (1e-2, 1e2),
                (1e-2, 1e2),
                (1e-2, 1e2),
            ],
        )
        return opt_res.x

    def loglik(
        self, pi0=0.2, eps=1e-3, lambda0=1.0, alpha=2.0, a0=1.0, b0=1.0, a1=0.5, b1=0.5
    ):
        """Compute the log-likelihood of the data under the current model parameters."""
        logll = 0.0
        for k in self.chrom_pos:
            _, _, ll = forward_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos[k],
                pi0=pi0,
                eps=eps,
                lambda0=lambda0,
                alpha=alpha,
                a0=a0,
                b0=b0,
                a1=a1,
                b1=b1,
            )
            logll += ll
        return logll

    def viterbi_algorithm(
        self, pi0=0.2, eps=1e-3, lambda0=1.0, alpha=2.0, a0=1.0, b0=1.0, a1=0.5, b1=0.5
    ):
        """Estimate the maximum-likelihood path through the HVR states.

        Returns a per-chromosome dictionary of the viterbi path under a specific parameterization.
        """
        path_dict = {}
        for k in self.chrom_pos:
            path, _, _ = viterbi_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos[k],
                pi0=pi0,
                eps=eps,
                lambda0=lambda0,
                alpha=alpha,
                a0=a0,
                b0=b0,
                a1=a1,
                b1=b1,
            )
            path_dict[k] = path
        return path_dict

    def forward_backward(
        self, pi0=0.2, eps=1e-3, lambda0=1.0, alpha=2.0, a0=1.0, b0=1.0, a1=0.5, b1=0.5
    ):
        """Run the forward-backward algorithm to estimate the posterior probability of HVR states.

        Returns a per-chromosome dictionary of the
        """
        gamma_dict = {}
        for k in self.chrom_pos:
            alphas, _, _ = forward_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos[k],
                pi0=pi0,
                eps=eps,
                lambda0=lambda0,
                alpha=alpha,
                a0=a0,
                b0=b0,
                a1=a1,
                b1=b1,
            )
            betas, _, _ = backward_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos[k],
                pi0=pi0,
                eps=eps,
                lambda0=lambda0,
                alpha=alpha,
                a0=a0,
                b0=b0,
                a1=a1,
                b1=b1,
            )
            gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
            gamma_dict[k] = gammas
        return gamma_dict
