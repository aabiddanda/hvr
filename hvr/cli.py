"""CLI for HVR detection."""
import gzip as gz
import logging
import pickle
import sys

import click
import numpy as np
import pandas as pd

from hvr import HVR

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--vcf",
    "-v",
    required=True,
    type=click.Path(exists=True),
    help="Input data VCF File.",
)
@click.option(
    "--window_size",
    "-w",
    required=True,
    default=100,
    type=int,
    help="Window size for segregating site estimation.",
)
@click.option(
    "--viterbi",
    is_flag=True,
    required=False,
    default=False,
    show_default=True,
    type=bool,
    help="Apply the Viterbi algorithm for determining hyper-variable regions.",
)
@click.option(
    "--chrom",
    required=False,
    default=None,
    show_default=True,
    type=list,
    help="Apply the algorithm only on a specific chromosome.",
)
@click.option(
    "--algo",
    required=False,
    default="Powell",
    type=click.Choice(["Nelder-Mead", "L-BFGS-B", "Powell"]),
    show_default=True,
    help="Method for numerical optimization in parameter inference.",
)
@click.option(
    "--recomb_map",
    "-r",
    required=False,
    default=1e-8,
    show_default=True,
    help="Recombination map",
)
@click.option(
    "--threads",
    "-t",
    required=False,
    type=int,
    default=2,
    help="Number of threads for reading in VCF file",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="karyohmm",
    help="Output file prefix.",
)
def main(
    vcf,
    viterbi,
    window_size=100,
    chrom=None,
    algo="Powell",
    recomb_map=1e-4,
    threads=2,
    out="hvr.results.gz",
):
    """HVR CLI."""
    logging.info(f"Starting to read input data {vcf}.")
    hvr = HVR(vcf_file=vcf)
    hvr.generate_window_data(
        window_size=window_size, threads=threads, chroms=chrom, strict_gt=True
    )
    hvr.interpolate_rec_dist(rec_rate=recomb_map)
    logging.info(f"Finished reading in {vcf}.")
    logging.info("Starting null parameter estimation ... ")
    hvr.est_lambda0()
    hvr.est_beta_params()
    logging.info(f"Estimated null rate of variants: {hvr.lambda0}")
    logging.info(f"Estimated null callrate parameters: {hvr.a0}, {hvr.b0}")
    logging.info("Finished null parameter estimation!")
    logging.info("Estimating alternative parameters ... ")
    [alpha, a1] = hvr.optimize_params(algo=algo)
    logging.info(f"Estimated alternative rate of variants: {alpha}")
    logging.info(f"Estimated alternative callrate parameters: {a1}, {a1}")
    logging.info("Finished estimating alternative parameters!")
    logging.info("Running forward-backward algorithm ...")
    gammas_dict = hvr.forward_backward(
        lambda0=hvr.lambda0, alpha=alpha, a0=hvr.a0, b0=hvr.b0, a1=a1, b1=a1
    )
    logging.info("Finished forward-backward algorithm!")
    logging.info(f"Writing output to {out}")
    res_dict = {
        "gammas": gammas_dict,
        "pos": hvr.chrom_pos,
        "lambda0": hvr.lambda0,
        "a0": hvr.a0,
        "b0": hvr.b0,
        "alpha": alpha,
        "a1": a1,
        "b1": a1,
    }
    with gz.open(out, "wb") as out_fp:
        pickle.dump(res_dict, out_fp)
    logging.info("Finished hvr analysis!")
