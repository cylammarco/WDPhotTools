# WDPhotTools

![example workflow name](https://github.com/cylammarco/WDPhotTools/workflows/Python%20package/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/cylammarco/WDPhotTools/badge.svg?branch=main)](https://coveralls.io/github/cylammarco/WDPhotTools?branch=main)
[![Documentation Status](https://readthedocs.org/projects/wdphottools/badge/?version=latest)](https://wdphottools.readthedocs.io/en/latest/?badge=latest)

This software can generate colour-colour diagram, colour-magnitude diagram in various photometric systems, plotting cooling profiles from different models, and compute theoretical white dwarf luminosity functions based on the built-in or supplied models of (1) initial mass function, (2) main sequence total lifetime, (3) initial-final mass relation, and (4) white dwarf cooling time.

the core parts of this work are three-fold: the first and the backbone of this work is the formatters that handle the output models from various works in the format as they are downloaded. This will allow the software to be updated with the newer models easily in the future. The second core part is the photometric fitter that solves for the WD parameters based on the photometry, with or without distance and reddening. The last part is to generate white dwarf luminosity function in bolometric magnitudes or in any of the photometric systems availalbe from the atmosphere model.


## Model Inspection

1. Plotting the WD cooling sequence in Gaia filters
The cooling models only include the modelling of the bolometric lumninosity, the synthetic photometry is not usually provided. We have included the synthetic colours computed by the [Montreal group](http://www.astro.umontreal.ca/~bergeron/CoolingModels/). By default, it maps the (logg, Mbol) to Gaia G band (DR3). The DA cooling tracks can be generated with

```python
from WDPhotTools import plotter

plotter.plot_atmosphere_model(invert_yaxis=True)
```
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/DA_cooling_tracks_with_plotter.png?raw=true)

or with finer control by using the interpolators 

```python
from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools.atmosphere_model_reader import atm_reader

atm = atm_reader()

# Default passband is G3
G = atm.interp_atm()
BP = atm.interp_atm(dependent="G3_BP")
RP = atm.interp_atm(dependent="G3_RP")

logg = np.arange(7.0, 9.5, 0.5)
Mbol = np.arange(0.0, 20.0, 0.1)

plt.figure(1, figsize=(8, 8))
for i in logg:
    logg_i = np.ones_like(Mbol) * i
    plt.plot(
        BP(logg_i, Mbol) - RP(logg_i, Mbol),
        G(logg_i, Mbol),
        label=r"$\log(g) = {}$".format(i),
    )

plt.ylim(20.0, 6.0)
plt.grid()
plt.legend()
plt.xlabel("(BP - RP) / mag")
plt.ylabel("G / mag")
plt.title("DA Cooling tracks")
plt.tight_layout()
```

![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/DA_cooling_tracks.png?raw=true)

2. Plotting the cooling models
The cooling sequence above is mostly concerned with the synthetic photometry, in this example, it is mostly regarding the physical properties beneath the photosphere. To check what models are embedded, you can use:

```
from WDPhotTools.cooling_model_reader import cm_reader

cr = cm_reader()
cr.list_cooling_model()
```
This should output:
```
Model: montreal_co_da_20, Reference: Bedard et al. 2020 CO DA
Model: montreal_co_db_20, Reference: Bedard et al. 2020 CO DB
Model: lpcode_he_da_07, Reference: Panei et al. 2007 He DA
Model: lpcode_co_da_07, Reference: Panei et al. 2007 CO DA
Model: lpcode_he_da_09, Reference: Althaus et al. 2009 He DA
Model: lpcode_co_da_10_z001, Reference: Renedo et al. 2010 CO DA Z=0.01
Model: lpcode_co_da_10_z0001, Reference: Renedo et al. 2010 CO DA Z=0.001
Model: lpcode_co_da_15_z00003, Reference: Althaus et al. 2015 DA Z=0.00003
Model: lpcode_co_da_15_z0001, Reference: Althaus et al. 2015 DA Z=0.0001
Model: lpcode_co_da_15_z0005, Reference: Althaus et al. 2015 DA Z=0.0005
Model: lpcode_co_db_17_z00005, Reference: Althaus et al. 2017 DB Y=0.4
Model: lpcode_co_db_17_z0001, Reference: Althaus et al. 2017 DB Y=0.4
Model: lpcode_co_db_17, Reference: Camisassa et al. 2017 DB
Model: basti_co_da_10, Reference: Salari et al. 2010 CO DA
Model: basti_co_db_10, Reference: Salari et al. 2010 CO DB
Model: basti_co_da_10_nps, Reference: Salari et al. 2010 CO DA, no phase separation
Model: basti_co_db_10_nps, Reference: Salari et al. 2010 CO DB, no phase separation
Model: lpcode_one_da_07, Reference: Althaus et al. 2007 ONe DA
Model: lpcode_one_da_19, Reference: Camisassa et al. 2019 ONe DA
Model: lpcode_one_db_19, Reference: Camisassa et al. 2019 ONe DB
Model: mesa_one_da_18, Reference: Lauffer et al. 2018 ONe DA
Model: mesa_one_db_18, Reference: Lauffer et al. 2018 ONe DB
```

And once you have picked a model, you can inspect what parameters are available with the model by supplying the model name, for example:

```python
cr.list_cooling_parameters('basti_co_da_10')
```
which will print
```
Available WD mass: [0.54 0.55 0.61 0.68 0.77 0.87 1.   1.1  1.2 ]
Parameter: log(Age), Column Name: age, Unit: (Gyr)
Parameter: Mass, Column Name: mass, Unit: M$_{\odot}$
Parameter: log(T$_{\mathrm{eff}})$, Column Name: Teff, Unit: (K)
Parameter: Luminosity, Column Name: lum, Unit: L$_{\odot}$
Parameter: u, Column Name: u, Unit: mag
Parameter: g, Column Name: g, Unit: mag
Parameter: r, Column Name: r, Unit: mag
Parameter: i, Column Name: i, Unit: mag
Parameter: z, Column Name: z, Unit: mag
```

Knowing which model and parameter you have access to, you can do basic visualisation using the default plotting funtion with the plotter again:
```python
from WDPhotTools import plotter

plotter.plot_cooling_model(mass=[0.2, 0.4, 0.6, 0.8, 1.0])
```
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/DA_cooling_model_from_plotter.png?raw=true)


## Photometric fitting
We provide 3 minimisers for fitting with synthetic photometry: `scipy.optimize.minimize`, `scipy.optimize.least_squares` and `emcee` (with the option to refine with a `scipy.optimize.minimize` in a bounded region.)

An example photometric fit of PSO J1801+6254 in 3 Gaia and 5 Pan-STARRS filters without providing a distance:
```python
ftr.fit(
    atmosphere="H",
    filters=["g_ps1", "r_ps1", "i_ps1", "z_ps1", "y_ps1", "G3", "G3_BP", "G3_RP"],
    mags=[21.1437, 19.9678, 19.4993, 19.2981, 19.1478, 20.0533, 20.7883, 19.1868],
    mag_errors=[0.0321, 0.0229, 0.0083, 0.0234, 0.0187, 0.006322, 0.118615, 0.070880],
    independent=["Mbol", "logg"],
    initial_guess=[15.0, 7.5],
    method='emcee',
    nwalkers=100,
    nsteps=5000,
    nburns=500
)
ftr.show_best_fit(display=False)
ftr.show_corner_plot(
    figsize=(10, 10),
    display=True,
    kwarg={
        "quantiles": [0.158655, 0.5, 0.841345],
        "show_titles": True,
        "truths": [3550, 7.45, 72.962],
    },
)

```
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/PSOJ1801p6254_emcee.png?raw=true)

And the associated corner plot where the blue line shows the true value. As we are not providing a distance in this case, as expected from the degeneracy between fitting distance and stellar radius (directly translate to logg in the fit), both truth values are well outside of the probability density maps in the corner plot:

![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/PSOJ1801p6254_emcee_corner.png?raw=true)

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
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/constant_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png?raw=true)

#### An example set of WDLFs with 1 Gyr of star burst
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png?raw=true)

#### An example set of WDLFs with a mean lifetime of 3 Gyr in the star formation rate
![alt text](https://github.com/cylammarco/WDPhotTools/blob/main/example/example_output/decay_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png?raw=true)
