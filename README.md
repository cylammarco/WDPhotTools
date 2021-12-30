# WDPhotTools

![example workflow name](https://github.com/cylammarco/WDPhotTools/workflows/Python%20package/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/cylammarco/WDPhotTools/badge.svg?branch=main)](https://coveralls.io/github/cylammarco/WDPhotTools?branch=main)

This software can generate colour-colour diagram, colour-magnitude diagram in various photometric systems, plotting cooling profiles from different models, and compute theoretical white dwarf luminosity functions based on the built-in or supplied models of (1) initial mass function, (2) main sequence total lifetime, (3) initial-final mass relation, and (4) white dwarf cooling time.

the core parts of this work are three-fold: the first and the backbone of this work is the formatters that handle the output models from various works in the format as they are downloaded. This will allow the software to be updated with the newer models easily in the future. The second core part is the photometric fitter that solves for the WD parameters based on the photometry, with or without distance and reddening. The last part is to generate white dwarf luminosity function in bolometric magnitudes or in any of the photometric systems availalbe from the atmosphere model.


## Model Inspection

### Plotting the WD cooling sequence in Gaia filters
The cooling models only include the modelling of the bolometric lumninosity, the synthetic photometry is not usually provided. We have included the synthetic colours computed by the [Montreal group](http://www.astro.umontreal.ca/~bergeron/CoolingModels/). By default, it maps the (logg, Mbol) to Gaia G band (DR3).

![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/DA_cooling_tracks.png?raw=true)

## Photometric fitting
An example photometric fit of PSO J1801+6254 in 3 Gaia and 5 Pan-STARRS filters.
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/PSOJ1801p6254.png?raw=true)

## Theoretical White Dwarf Luminosity Function

The options for the various models include:

### Initial Mass Function

1. Kroupa 2001
2. Charbrier 2003
3. Charbrier 2003 (including binary)
4. Manual

### Main Sequence Total Lifetime

1. Choi et al. 2016
2. Bressan et al. 2013 (solar metallicity)
3. Manual

to be added:

4. other metallicities
5. other MESA models

### Initial-Final Mass Relation

1. C08 - Catalan et al. 2008
2. C08b - Catalan et al. 2008 (two-part)
3. S09 - Salaris et al. 2009
4. S09b - Salaris et al. 2009 (two-part)
5. W09 - Williams, Bolte & Koester 2009
6. K09 - Kalirai et al. 2009
7. K09b - Kalirai et al. 2009 (two-part)
8. C18 - Cummings et al. 2018
9. EB18 - El-Badry et al. 2018
10. Manual

### White Dwarf cooling time

L/I/H are used to denote the availability in the low, intermediate and high mass models, where the dividing points are at 0.5 and 1.0 solar masses.

The brackets denote the core type/atmosphere type/mass range/other special properties.

1. Montreal models
    1. Bedard et al. 2020 -- LIH [CO/DA+DB/0.2-1.3]
2. LaPlata models
    1.  Panei et al. 2007 -- L [He+CO/DA/0.187-0.448]
    2. Althaus et al. 2009 -- L [He/DA/0.220-0.521]
    3. Renedo et al. 2010 -- I [CO/DA/0.505-0.934/Z=0.001-0.01]
    4. Althaus et al. 2015 -- I [CO/DA/0.506-0.826/Z=0.0003-0.001]
    5. Althaus et al. 2017 -- LI [CO/DA/0.434-0.838/Y=0.4]
    6. Camisassa et al. 2017 -- I [CO/DB/0.51-1.0]
    7. Althaus et al. 2007 -- H [ONe/DA/1.06-1.28]
    8. Camisassa et al. 2019 -- H [ONe/DA+B/1.10-1.29]
3. BaSTI models
    1. Salaris et al. 2010 -- IH [CO/DA+B/0.54-1.2/ps+nps]
4. MESA models
    1. Lauffer et al. 2018 -- H [CONe/DA+B/1.012-1.308]

#### An example set of WDLFs with constant star formation rate
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/constant_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png?raw=true)

#### An example set of WDLFs with 1 Gyr of star burst
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png?raw=true)

#### An example set of WDLFs with a mean lifetime of 3 Gyr in the star formation rate
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/decay_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png?raw=true)
