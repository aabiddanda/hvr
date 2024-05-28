"""CLI for HVR detection."""
import logging
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
    type=float,
    show_default=True,
    help="Recombination map",
)
@click.option(
    "--gzip",
    "-g",
    is_flag=True,
    required=False,
    type=bool,
    default=True,
    help="Gzip output files",
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
    chrom=None,
    algo="Powell",
    recomb_map=1e-8,
    gzip=True,
    threads=2,
    out="hvr",
):
    """HVR CLI."""
    logging.info(f"Starting to read input data {vcf}.")
    hvr = HVR(vcf_file=vcf)
    hvr.generate_window_data(
        window_size=100, threads=threads, chroms=chrom, strict_gt=True
    )
    hvr.interpolate_rec_dist(rec_rate=recomb_map)
    logging.info(f"Finished reading in {vcf}.")
    logging.info("Finished hvr analysis!")
    pass
