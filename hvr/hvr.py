"""Functions + classes useful for """

import warnings

import numpy as np
from cyvcf2 import VCF
from hvr_utils import *
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm


class HVR:
    """Class for HMM-based calling of hyper-variable regions in genomes."""

    def __init__(vcf_file):
        self.vcf_file = None
        self.pos = None
        self.nvars = None
        self.call_rate = None

    def generate_window_data(self, window_size=100):
        """Create the emission / position vectors at intervals of 100 basepairs."""
        assert window_size > 0
        if window_size <= 100:
            warnings.warn("Warning: window-size for estimation may be potentially too ")
        vcf = VCF(self.vcf_file, **kwargs)
        chroms = vcf.seqnames
        chrom_cnt_dict = {}
        chrom_call_rate_dict = {}
        chrom_pos_dict = {}
        for c in chroms:
            pos = []
            cnts = []
            call_rates = []
            cur_call_rate = []
            cnt = 0
            start = 0.0
            end = window_size
            for i, v in tqdm(enumerate(vcf(c))):
                if i == 0:
                    start = v.POS
                    end = v.POS + window_size
                if v.POS > end:
                    pos.append((start, end))
                    start = end
                    end = end + window_size
                    cnts.append(cnt)
                    cnt = 1
                    call_rates.append(cur_call_rate)
                    cur_call_rate = [v.call_rate]
                else:
                    cnt += 1
                    cur_call_rate.append(v.call_rate)
            assert len(cnts) == len(call_rates)
            chrom_pos_dict[c] = pos
            chrom_cnt_dict[c] = cnts
            chrom_call_rate_dict[c] = call_rates
        # Actually set the variants here ...
        self.chrom_cnts = chrom_cnt_dict
        self.chrom_pos = chrom_pos_dict
        self.chrom_call_rate = chrom_call_rate_dict

    def interpolate_rec_dist(self, recmap=None, rec_rate=1e-4):
        """Interpolate the recombination distance."""
        assert (rec_rate > 0) and (rec_rate < 1)
        if recmap is None:
            pass
        else:
            pass

    def est_beta_params(self):
        """Estimate the null beta parameters."""

    def optimize_params(self, algo="L-BFGS-B"):
        """Optimize the underlying parameters."""
        pass

    def forward_algorithm(self):
        """Run the forward algorithm under this model."""
        pass

    def viterbi_algorithm(self):
        """Estimate the maximum-likelihood path through the HVR states."""
        pass

    def forward_backward(self):
        """Run the forward-backward algorithm to estimate the posterior probability of HVR states."""
        pass
