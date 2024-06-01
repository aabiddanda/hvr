"""Class for defining HMM for identifying highly variable regions in nematode genomes."""

import warnings

import numpy as np
from cyvcf2 import VCF
from hvr_utils import backward_algo, forward_algo, viterbi_algo
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import logsumexp as logsumexp_sp
from tqdm import tqdm


class HVR:
    """Class for HMM-based calling of hyper-variable regions in genomes."""

    def __init__(self, vcf_file):
        """Initialize the HVR class."""
        self.vcf_file = vcf_file
        self.chrom_pos = None
        self.chrom_cnts = None
        self.chrom_call_rate = None

    def generate_window_data(self, window_size=100, chroms=None, **kwargs):
        """Create the emission / position vectors at intervals of 100 basepairs."""
        assert window_size > 0
        if window_size <= 100:
            warnings.warn("Warning: window-size for estimation is <= 100 bp!")
        vcf = VCF(self.vcf_file, **kwargs)
        if chroms is None:
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
                pos = np.array(pos)
                bins = np.arange(np.min(pos), np.max(pos), window_size)
                call_rate = np.array(call_rate)
                call_rates_test = np.zeros(bins.size)
                cnts = np.zeros(bins.size, dtype=int)
                idx = np.digitize(pos, bins=bins)
                for i in tqdm(range(bins.size)):
                    call_rates_test[i] = (
                        np.mean(call_rate[idx == i])
                        if call_rate[idx == i].size > 0
                        else 1.0
                    )
                    cnts[i] = int(pos[idx == i].size)
                call_rates = call_rates_test
                bins = np.insert(bins, 0, 0)
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
                self.chrom_pos_scaled[k] = self.chrom_pos[k] * rec_rate
        else:
            raise NotImplementedError(
                "Interpolation via a recombination map is not currently supported!"
            )

    def est_beta_params(self):
        """Estimate the null beta parameters across all the call-rates.

        NOTE: this is just using moment-matching for the beta distribution.
        """
        xs = np.hstack([self.chrom_call_rate[i] for i in self.chrom_call_rate])
        mu = np.nanmean(xs[(xs < 1.0) & (xs > 0.0)])
        sigma2 = np.nanvar(xs)
        a = -(sigma2 + mu**2 - mu) / sigma2
        b = ((sigma2 + mu**2 - mu) * (mu - 1)) / sigma2
        self.a0 = a
        self.b0 = b

    def est_lambda0(self):
        """Estimate the rate of variants."""
        xs = np.hstack([self.chrom_call_rate[i] for i in self.chrom_call_rate])
        self.lambda0 = np.nanmean(xs)

    def optimize_params(self, algo="L-BFGS-B"):
        """Optimize the parameters."""
        assert self.lambda0 is not None
        assert self.a0 is not None
        assert self.b0 is not None
        assert algo in ["L-BFGS-B", "Powell"]
        opt_res = minimize(
            lambda x: -self.loglik(
                lambda0=self.lambda0,
                alpha=x[0],
                a0=self.a0,
                b0=self.b0,
                a1=x[1],
                b1=x[1],
            ),
            x0=[5.0, 2.0],
            method=algo,
            bounds=[
                (2.0, 100.0),
                (1e-2, 1e1),
            ],
            tol=1e-4,
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
        )
        return opt_res.x

    def loglik(self, lambda0=1.0, alpha=2.0, a0=1.0, b0=1.0, a1=0.5, b1=0.5):
        """Compute the log-likelihood of the data under the current model parameters."""
        logll = 0.0
        for k in self.chrom_pos:
            _, _, ll = forward_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos_scaled[k],
                lambda0=lambda0,
                alpha=alpha,
                a0=a0,
                b0=b0,
                a1=a1,
                b1=b1,
            )
            logll += ll
        return logll

    def viterbi_algorithm(self, alpha=2.0, a1=0.5, b1=0.5):
        """Estimate the maximum-likelihood path through the HVR states.

        Returns a per-chromosome dictionary of the viterbi path under a specific parameterization.
        """
        path_dict = {}
        for k in self.chrom_pos:
            path, _, _ = viterbi_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos_scaled[k],
                lambda0=self.lambda0,
                alpha=alpha,
                a0=self.a0,
                b0=self.b0,
                a1=a1,
                b1=b1,
            )
            path_dict[k] = path
        return path_dict

    def forward_backward(self, lambda0=1.0, alpha=2.0, a0=1.0, b0=1.0, a1=0.5, b1=0.5):
        """Run the forward-backward algorithm to estimate the posterior probability of HVR states.

        Returns a per-chromosome dictionary of the
        """
        gamma_dict = {}
        for k in self.chrom_pos:
            alphas, _, _ = forward_algo(
                cnts=self.chrom_cnts[k],
                call_rates=self.chrom_call_rate[k],
                pos=self.chrom_pos_scaled[k],
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
                pos=self.chrom_pos_scaled[k],
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
