![IVM](http://ivm.vu.nl/en/Images/IVM_logo_rgb2_tcm234-851594.svg)

# MRIA Argentina
This repository holds the code to perform a multiregional impact assessment for Argentina.

## Python requirements
Recommended option is to use a [miniconda](https://conda.io/miniconda.html)
environment to work in for this project, relying on conda to handle some of the
trickier library dependencies.

```bash
# Add conda-forge channel for extra packages
conda config --add channels conda-forge

# Create a conda environment for the project and install packages
conda env create -f .environment.yml
```

## How to run the model from a notebook

Move to the project folder (**MRIA_Argentina**), activate the virtual environment and jupyter lab:

```bash
(base) conda activate MRIA_ARG
(MRIA_ARG) jupyter lab
```
## GAMS dependency for the MRIA model
When using the model with one of the predefined data sources, a GAMS dependency is not necessary. When using 
new data, a new file with marginal values of rationing is required. This can (for now) only be done through a licensed GAMS version. 

## Source data:

**INDEC** : https://www.indec.gob.ar --> Cuadros de oferta y utilización (COU)

**OECD** : http://www.oecd.org/sti/ind/input-outputtables.htm 

**EORA** : https://worldmrio.com/countrywise/ 

**GTAP** : https://www.gtap.agecon.purdue.edu/ 

## Background papers

```
Koks, E. E., & Thissen, M. (2016). A multiregional impact assessment model for disaster analysis. 
                                    Economic Systems Research, 28(4), 429-449.
```

## License
Copyright (C) 2020 Elco Koks. All versions released under the [MIT license](LICENSE.md).
