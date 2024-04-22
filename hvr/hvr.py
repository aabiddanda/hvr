"""Functions + classes useful for """

import numpy as np
from cyvcf2 import VCF
from hvr_utils import *
from scipy.optimize import minimize


class HVR:
    def __init__(vcf_file):
        pass

    def generate_window_data(self, window_size=100):
        """Create the emission / position vectors at intervals of 100 basepairs."""
        pass

    def optimize_params(self):
        """Optimize parameters for detection."""
        pass

    def forward_algorithm(self):
        """Conduct inference under the forward algorithm."""
        pass

    def viterbi_algorithm(self):
        """Estimate the maximum-likelihood path through the HVR states."""
        pass

    def forward_backward(self):
        """Run the forward-backward algorithm."""
        pass
