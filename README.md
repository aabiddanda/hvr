# `hvr`: a method for detection of hyper-variable regions in nematode genomes

## Method

The core idea behind the method is to provide a model-driven method to isolate "hyper-divergent" loci from a collection of nematode individuals. The model is broader and can be applied to a larger number of contexts as well.

Briefly, there are two key features that are contained within hyper-divergent regions: 1) a high local rate of polymorphism and/or 2) a large proportion of unalignable reads leading to "streaks" of missingness. The first feature indicates a higher "rate" of polymorphism compared to the genome-wide background. The second feature means that there may be a high-proportion of variants locally that simply have missing data (due to lack of alignment).

We choose to model these jointly within a haplotype-copying model framework, where each


## Installation

```
git clone
cd hvr
pip install .
```

## Running

In addition to the baseline model, we have also allowed for testing several new


## Contact

Arjun Biddanda (@aabiddanda)
