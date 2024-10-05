# `hvr`: a method for detection of hyper-variable regions in nematode genomes

## Method

The core idea behind the method is to provide a model-driven method to isolate "hyper-divergent" loci from a collection of nematode individuals. The model is broader and can be applied to a larger number of contexts as well.

Briefly, there are two key features that are contained within hyper-divergent regions: 1) a high local rate of polymorphism and/or 2) a large proportion of unalignable reads leading to "streaks" of missingness. The first feature indicates a higher "rate" of polymorphism compared to the genome-wide background. The second feature means that there may be a high-proportion of variants locally that simply have missing data (due to lack of alignment).

We choose to model these jointly within an HMM framework with a joint emission model.  

## Installation

```
git clone
cd hvr
pip install .
```

## Running

We have implemented a CLI for this as well `hvr-cli`, which has the following options:

```
Usage: hvr-cli [OPTIONS]

  HVR CLI.

Options:
  -v, --vcf PATH                  Input data VCF File.  [required]
  -w, --window_size INTEGER       Window size for segregating site estimation.
                                  [required]
  --chrom TEXT                    Apply the HMM on this specific chromosome.
                                  [default: I]
  --algo [Nelder-Mead|L-BFGS-B|Powell]
                                  Method for numerical optimization in
                                  parameter inference.  [default: Powell]
  -r, --recomb_map FLOAT          Recombination map  [default: 1e-08]
  -t, --threads INTEGER           Number of threads for reading in VCF file
  -o, --out TEXT                  Output file prefix.  [required]
  --help                          Show this message and exit.
```

The call for identifying hyper-variable regions for a specific chromosome is therefore: 

`hvr-cli -v {input.vcf} -w {params.window_size} --threads {threads} --chrom {wildcards.chrom} -r {params.rec_rate} -o {output.pkl}`

which will both learn the null parameters via MLE as well as apply the forward-backward algorithm to identify regions with a high posterior probability of being a HVR.

## Contact

Arjun Biddanda (@aabiddanda)
