# WDLFBuilder

This software constructs theoretical white dwarf luminosity functions based on the built-in or supplied models of (1) initial mass function, (2) main sequence total lifetime, (3) initial-final mass relation, and (4) white dwarf cooling time.

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

1. 'montreal_thick' - Montreal hydrogen atmospheremodel (2020)
2. 'montreal_thin' - Montreal helium atmosphere model (2020)
3. 'laplata' - La Plata model (2000)

to be added:

4. 'basti' - BASTI model (2000)
5. 'bastips' - BASTI model with phase separation (2000)

## An example set of WDLFs with constant star formation rate
![alt text](https://github.com/cylammarco/WDLFBuilder/blob/main/wdlf_constant_sfr.png?raw=true)
