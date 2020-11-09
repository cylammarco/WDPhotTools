import io
import glob
import numpy as np


def althaus07_formatter():
    '''
    A formatter to load the Althaus et al. 2007 WD cooling model

    '''

    filelist = glob.glob('wd_cooling/althaus07/*.dat')

    # Prepare the array column dtype
    column_key = np.array(('lum', 'logg', 'B-V', 'V-R', 'V-K', 'R-I', 'J-H',
                           'H-K', 'V-I', 'U-V', 'BC', 'dmag_v', 'age'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('_')[-1][:3]
                     for i in filelist]).astype(np.float64) / 100000.

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] = 10.**cooling_model[i]['age']

    return mass, cooling_model


def althaus09_formatter(mass_range='all'):
    '''
    A formatter to load the Althaus et al. 2009 WD cooling model

    '''

    filelist = glob.glob('wd_cooling/althaus09/z.*')

    # Prepare the array column dtype
    column_key = np.array(('Teff', 'logg', 'lum', 'age', 'BC', 'M_V', 'U', 'B',
                           'V', 'R', 'I', 'J', 'H', 'K', 'L', 'U-B', 'B-V',
                           'V-K', 'V-I', 'R-I', 'J-H', 'H-K', 'K-L'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('.')[-2]
                     for i in filelist]).astype(np.float64) / 100000.

    if mass_range == 'all':
        pass
    if mass_range == 'low':
        mask_low = mass < 0.5
        mass = mass[mask_low]
        filelist = np.array(filelist)[mask_low]

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] *= 1E9

    return mass, cooling_model


def althaus15_formatter(model):
    '''
    A formatter to load the Althaus et al. 2009 WD cooling model

    '''

    # Z=0.00003 models
    if model == 'lpcode_co_da_15_z00003':
        filelist = glob.glob('wd_cooling/althaus15/Z=3d-5/*.trk')

    # Z=0.0001 models
    if model == 'lpcode_co_da_15_z0001':
        filelist = glob.glob('wd_cooling/althaus15/Z=1d-4/*.trk')

    # Z=0.0005 models
    if model == 'lpcode_co_da_15_z0005':
        filelist = glob.glob('wd_cooling/althaus15/Z=5d-4/*.trk')

    # Prepare the array column dtype
    column_key = np.array(
        ('lum', 'Teff', 'Tc', 'Roc', 'Hc', 'Hec', 'Con_s', 'Con_c', 'age',
         'mass', 'mdot', 'model_no', 'Lpp', 'Lcno', 'LHe', 'LCC', 'dSdt',
         'Lnu', 'MHtot', 'HeBuf', 'mass_Hfc', 'mass_Hefc', 'logg', 'Rsun',
         'LH', 'ps'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('.')[-2][-5:]
                     for i in filelist]).astype(np.float64) / 100000.

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] = 10.**cooling_model[i]['age'] * 1E6
        cooling_model[i]['age'] -= min(cooling_model[i]['age'])

    return mass, cooling_model


def althaus17_formatter(model):
    '''
    A formatter to load the Althaus et al. 2009 WD cooling model

    '''

    # Y=0.4, Z=0.001 models
    if model == 'lpcode_co_db_17_z00005':
        filelist = glob.glob('wd_cooling/althaus17/*d4.trk')

    # Y=0.4, Z=0.0005 models
    if model == 'lpcode_co_db_17_z0001':
        filelist = glob.glob('wd_cooling/althaus17/*d3.trk')

    # Prepare the array column dtype
    column_key = np.array(
        ('lum', 'Teff', 'Tc', 'Roc', 'Hc', 'Hec', 'Con_s', 'Con_c', 'age',
         'mass', 'mdot', 'model_no', 'Lpp', 'Lcno', 'LHe', 'LCC', 'dSdt',
         'Lnu', 'MHtot', 'HeBuf', 'mass_Hfc', 'mass_Hefc', 'logg', 'Rsun',
         'LH', 'ps'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('/')[-1].split('_')[0]
                     for i in filelist]).astype(np.float64)
    wd_mass = np.zeros_like(mass)

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        wd_mass[i] = cooling_model[i]['mass']
        cooling_model[i]['age'] = 10.**cooling_model[i]['age'] * 1E6
        cooling_model[i]['age'] -= min(cooling_model[i]['age'])

    if mass_range == 'all':
        pass
    if mass_range == 'low':
        mask_low = mass < 0.5
        wd_mass = wd_mass[mask_low]
        cooling_model = cooling_model[mask_low]
    if mass_range == 'intermediate':
        mask_intermediate = (mass >= 0.5) & (mass <= 1.0)
        wd_mass = wd_mass[mask_intermediate]
        cooling_model = cooling_model[mask_intermediate]

    return wd_mass, cooling_model


def bedard20_formatter(model, mass_range='all'):
    '''
    A formatter to load the Bedard et al. 2020 WD cooling model from
    http://www.astro.umontreal.ca/~bergeron/CoolingModels/

    The thick and thin models are for DA and DB WD, respectively.

    '''

    # DA models
    if model == 'montreal_co_da_20':
        filelist = glob.glob('wd_cooling/bedard20/*thick*')

    # DB models
    if model == 'montreal_co_db_20':
        filelist = glob.glob('wd_cooling/bedard20/*thin*')

    # Prepare the array column dtype
    column_key = np.array(
        ('step', 'Teff', 'logg', 'rayon', 'age', 'lum', 'logTc', 'logPc',
         'logrhoc', 'MxM', 'logqx', 'lumnu', 'logH', 'logHe', 'logC', 'logO'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('_')[2]
                     for i in filelist]).astype(np.float64) / 100.

    if mass_range == 'all':
        pass
    if mass_range == 'low':
        mask_low = mass < 0.5
        mass = mass[mask_low]
        filelist = np.array(filelist)[mask_low]
    if mass_range == 'intermediate':
        mask_intermediate = (mass >= 0.5) & (mass <= 1.0)
        mass = mass[mask_intermediate]
        filelist = np.array(filelist)[mask_intermediate]
    if mass_range == 'high':
        mask_high = mass > 1.0
        mass = mass[mask_high]
        filelist = np.array(filelist)[mask_high]

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        with open(filepath) as infile:

            count = -5
            cooling_model_text = ''
            for line_i in infile:

                count += 1

                if count <= 0:
                    continue

                if count % 3 != 0:
                    cooling_model_text += line_i.rstrip('\n')
                else:
                    cooling_model_text += line_i

        cooling_model[i] = np.loadtxt(io.StringIO(cooling_model_text),
                                      dtype=dtype)

    return mass, cooling_model


def camisassa17_formatter():
    '''
    A formatter to load the Camisassa et al. 2017 WD cooling model

    The progenitor lifetime is taken off based on the extrapolation from
    Table 1
    https://iopscience.iop.org/article/10.3847/0004-637X/823/2/158

    '''

    # Y=0.4, Z=0.0005 models
    filelist = glob.glob('wd_cooling/camisassa17/*.trk')

    # Prepare the array column dtype
    column_key = np.array(
        ('lum', 'Teff', 'Tc', 'Roc', 'Hc', 'Hec', 'Con_s', 'Con_c', 'age',
         'mass', 'mdot', 'model_no', 'Lpp', 'Lcno', 'LHe', 'LCC', 'logG',
         'Lnu', 'MHtot', 'HeBuf', 'mass_Hfc', 'mass_Hefc', 'logg', 'Rsun'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('/')[-1][:3]
                     for i in filelist]).astype(np.float64) / 100.

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] = 10.**cooling_model[i]['age'] * 1E6
        cooling_model[i]['age'] -= min(cooling_model[i]['age'])

    return mass, cooling_model


def camisassa19_formatter(model):
    '''
    A formatter to load the Camisassa et al. 2019 ultramassive WD cooling model

    Some columns populated with 'I' are replaced with the nearest values.

    '''

    # DA model
    if model == 'lpcode_one_da_19':
        filelist = glob.glob('wd_cooling/camisassa19/*hrich.dat')

    # DB model
    if model == 'lpcode_one_db_19':
        filelist = glob.glob('wd_cooling/camisassa19/*hdef.dat')

    # Prepare the array column dtype
    column_key = np.array(
        ('lum', 'Teff', 'Tc', 'Roc', 'Hc', 'Hec', 'Con_s', 'Con_c', 'age',
         'mass', 'mdot', 'Lnu', 'MHtot', 'logg', 'Rsun', 'LH', 'sf'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('/')[-1][:3]
                     for i in filelist]).astype(np.float64) / 100.

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=2, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] = 10.**cooling_model[i]['age'] * 1E6
        cooling_model[i]['age'] -= min(cooling_model[i]['age'])

    return mass, cooling_model


def lauffer18_formatter(model):
    '''
    A formatter to load the Lauffer et al. 2018 WD cooling model

    '''

    # H models
    if model == 'mesa_one_da_18':
        filelist = glob.glob('wd_cooling/lauffer18/H_*.dat')

    # He models
    if model == 'mesa_one_db_18':
        filelist = glob.glob('wd_cooling/lauffer18/He_*.dat')

    # Prepare the array column dtype
    column_key = np.array(
        ('Teff', 'lum', 'logg', 'Rsun', 'mass', 'age', 'total_age'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('-M')[-1][:-4]
                     for i in filelist]).astype(np.float64)

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] *= 1E9

    return mass, cooling_model


def panei07_formatter(model):
    '''
    A formatter to load the Panei et al. 2007 WD cooling model

    '''

    # He core models
    if model == 'lpcode_he_da_07':
        filelist = glob.glob('wd_cooling/panei07/*He.SDSS')

    # CO core models
    if model == 'lpcode_co_da_07':
        filelist = glob.glob('wd_cooling/panei07/*CO.SDSS')

    # Prepare the array column dtype
    column_key = np.array(
        ('Teff', 'logg', 'lum', 'age', 'u', 'g', 'r', 'i', 'z'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('.')[-2][:5]
                     for i in filelist]).astype(np.float64) / 100000.

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] *= 1E9

    return mass, cooling_model


def renedo10_formatter(model):
    '''
    A formatter to load the Renedo et al. 2010 WD cooling model from
    http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html

    Two metallicity for DA are available: Z=0.01 and Z=0.001

    '''

    # Solar metallicity model
    if model == 'lpcode_co_da_10':
        filelist = glob.glob('wd_cooling/renedo10/*z001.trk')

    # Low metallicity model
    if model == 'lpcode_co_da_10_z0001':
        filelist = glob.glob('wd_cooling/renedo10/*z0001.trk')

    # Prepare the array column dtype
    column_key = np.array(
        ('lum', 'Teff', 'logTc', 'logrhoc', 'age', 'mass', 'lumpp', 'lumcno',
         'lumhe', 'lumnu', 'logH', 'logg', 'rsun'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('_')[1][-4:]
                     for i in filelist]).astype(np.float64) / 1000.

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] *= 1E6

    return mass, cooling_model


def salaris10_formatter(model, mass_range):
    '''
    A formatter to load the Salaris et al. 2010 WD cooling model from

    '''

    # DA model with phase separation
    if model == 'basti_co_da_10':
        filelist = glob.glob('wd_cooling/salaris10/*DAsep.sdss')

    # DB model with phase separation
    if model == 'basti_co_db_10':
        filelist = glob.glob('wd_cooling/salaris10/*DBsep.sdss')

    # DA model without phase separation
    if model == 'basti_co_da_10_nps':
        filelist = glob.glob('wd_cooling/salaris10/*DAnosep.sdss')

    # DB model without phase separation
    if model == 'basti_co_db_10_nps':
        filelist = glob.glob('wd_cooling/salaris10/*DBnosep.sdss')

    # Prepare the array column dtype
    column_key = np.array(
        ('age', 'mass', 'Teff', 'lum', 'u', 'g', 'r', 'i', 'z'))
    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Get the mass from the file name
    mass = np.array([i.split('COOL')[-1][:3]
                     for i in filelist]).astype(np.float64) / 100.

    if mass_range == 'all':
        pass
    if mass_range == 'intermediate':
        mask_intermediate = (mass >= 0.5) & (mass <= 1.0)
        mass = mass[mask_intermediate]
        filelist = np.array(filelist)[mask_intermediate]
    if mass_range == 'high':
        mask_high = (mass > 1.0)
        mass = mass[mask_high]
        filelist = np.array(filelist)[mask_high]

    # Create an empty array for holding the cooling models
    cooling_model = np.array(([''] * len(mass)), dtype='object')

    for i, filepath in enumerate(filelist):

        cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

        # Convert the luminosity into erg/s
        cooling_model[i]['lum'] = 10.**cooling_model[i]['lum'] * 3.826E33

        # Convert the age to yr
        cooling_model[i]['age'] = 10.**cooling_model[i]['age']

    return mass, cooling_model
