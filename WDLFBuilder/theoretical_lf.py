import numpy as np
from scipy import optimize, integrate
from scipy.interpolate import interp1d
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib import pyplot as plt
import warnings

from .model_reader import *


class WDLF:
    '''
    Computing the theoretical WDLFs based on the input IFMR, WD cooling and
    MS lifetime models.

    We are using little m for WD mass and big M for MS mass throughout this
    package.

    All the models are reporting in different set of units. They are all
    converted by the formatter to this set of units: (1) mass is in solar mass,
    (2) luminosity is in erg/s, (3) time/age is in year.

    For conversion, we use (1) M_sun = 1.98847E30 and (2) L_sun = 3.826E33.

    '''
    def __init__(self,
                 imf_model='C03',
                 ifmr_model='EB18',
                 low_mass_cooling_model='montreal_co_da_20',
                 intermediate_mass_cooling_model='montreal_co_da_20',
                 high_mass_cooling_model='montreal_co_da_20',
                 ms_model='C16'):
        # The IFMR, WD cooling and MS lifetime models are required to
        # initialise the object.
        self.set_imf_model(imf_model)
        self.set_ifmr_model(ifmr_model)
        self.set_low_mass_cooling_model(low_mass_cooling_model)
        self.set_intermediate_mass_cooling_model(
            intermediate_mass_cooling_model)
        self.set_high_mass_cooling_model(high_mass_cooling_model)
        self.set_ms_model(ms_model)

        self.cooling_interpolator = None

    def _imf(self, M):
        '''
        Compute the initial mass function based on the pre-selected IMF model
        and the given mass (M).

        See set_imf_model() for more details.

        Parameters
        ----------
        M: float, list of float or array of float
            Input MS mass

        Returns
        -------
        MF: array
            Array of MF, normalised to 1 at 1M_sun.

        '''

        M = np.asarray(M).reshape(-1)

        if self.imf_model == 'K01':

            MF = M**-2.3

            # mass lower than 0.08 is impossible, so that range is ignored.
            if (M < 0.5).any():

                M_mask = M < 0.5
                MF[M_mask] = M[M_mask]**1.3

        elif self.imf_model == 'C03':

            MF = M**-2.3
            if (M < 1).any():
                M_mask = np.array(M < 1.).reshape(-1)
                # 0.158 / (ln(10) * M) = 0.06861852814 / M
                # log(0.079) = -1.1023729087095586
                # 2 * 0.69**2. = 0.9522
                MF[M_mask] = (0.06861852814 / M[M_mask]) * np.exp(
                    -(np.log10(M[M_mask]) + 1.1023729087095586)**2. / 0.9522)

        elif self.imf_model == 'C03b':

            MF = M**-2.3

            if (M <= 1).any():
                M_mask = np.array(M <= 1.).reshape(-1)
                # 0.086 * 1. / (ln(10) * M) = 0.03734932544 / M
                # log(0.22) = -0.65757731917
                # 2 * 0.57**2. = 0.6498
                MF[M_mask] = (0.03734932544 / M[M_mask]) * np.exp(
                    -(np.log10(M[M_mask]) + 0.65757731917)**2. / 0.6498)

        elif self.imf_model == 'manual':

            MF = self.imf_function(M)

        else:

            raise ValueError(
                'Please choose from K01, C03 and C03b. Use set_imf_model() to '
                'change to a valid model.')

        return MF

    def _ms_age(self, M):
        '''
        Compute the main sequence lifetime based on the pre-selected MS model
        and the given mass (M).

        See set_ms_model() for more details.

        Parameters
        ----------
        M: float, list of float or array of float
            Input MS mass

        Returns
        -------
        age: array
            Array of total MS lifetime, same size as M.

        '''

        M = np.asarray(M).reshape(-1)

        if self.ms_model == 'Bressan':
            datatable = np.loadtxt('ms_lifetime/bressan00170279.csv',
                                   delimiter=',')
            massi = np.array(datatable[:, 0]).astype(np.float64)
            time = np.array(datatable[:, 1]).astype(np.float64)
            age = interp1d(massi, time)(M)

        elif self.ms_model == 'C16':
            age = 10**(13.37807 - 6.292517 * M + 4.451837 * M**2 -
                       1.773315 * M**3 + 0.2944963 * M**4)
            if (M > 2.11).any():
                age[M > 2.11] = 10**(10.75941 - 1.043523 * M[M > 2.11] +
                                     0.1366088 * M[M > 2.11]**2 -
                                     7.110290e-3 * M[M > 2.11]**3)

        elif self.ms_model == 'manual':

            age = self.ms_function(M)

        else:
            raise ValueError('Please choose from a valid MS model.')

        return age

    def _ifmr(self, M, fill_value=0):
        '''
        Compute the final mass (i.e. WD mass) based on the pre-selected IFMR
        model and the zero-age MS mass (M).

        See set_ifmr_model() for more details.

        Parameters
        ----------
        M: float, list of float or array of float
            Input MS mass
        fill_value: numeric, str or list of size 2 (Default: 0)
            Value to fill if m_WD is outside the interpolated grid. Set to
            'extrapolate' to return the extrapolated values, for 'C18' and
            'EB18' only.

        Returns
        -------
        m: array
            Array of WD mass, same size as M.

        '''

        M = np.asarray(M).reshape(-1)

        if self.ifmr_model == 'C08':
            m = 0.117 * M + 0.384
            if (m < 0.4349).any():
                m[m < 0.4349] = 0.4349

        elif self.ifmr_model == 'C08b':
            m = 0.096 * M + 0.429
            if (M >= 2.7).any():
                m[M >= 2.7] = 0.137 * M[M >= 2.7] + 0.318
            if (m < 0.4746).any():
                m[m < 0.4746] = 0.4746

        elif self.ifmr_model == 'S09':
            m = 0.084 * M + 0.466
            if (m < 0.5088).any():
                m[m < 0.5088] = 0.5088

        elif self.ifmr_model == 'S09b':
            m = 0.134 * M[M < 4.0] + 0.331
            if (M >= 4.0).any():
                m = 0.047 * M[M >= 4.0] + 0.679

            if (m < 0.3823).any():
                m[m < 0.3823] = 0.3823

        elif self.ifmr_model == 'W09':
            m = 0.129 * M + 0.339
            if (m < 0.3893).any():
                m[m < 0.3893] = 0.3893

        elif self.ifmr_model == 'K09':
            m = 0.109 * M + 0.428
            if (m < 0.4804).any():
                m[m < 0.4804] = 0.4804

        elif self.ifmr_model == 'K09b':
            m = 0.101 * M + 0.463
            if (m < 0.4804).any():
                m[m < 0.4804] = 0.4804

        elif self.ifmr_model == 'C18':
            m = interp1d((0.23, 0.5, 0.95, 2.8, 3.65, 8.2, 10),
                         (0.19, 0.4, 0.50, 0.72, 0.87, 1.25, 1.4),
                         fill_value='extrapolate',
                         bounds_error=False)(M)

        elif self.ifmr_model == 'EB18':
            m = interp1d((0.95, 2.75, 3.54, 5.21, 8.),
                         (0.5, 0.67, 0.81, 0.91, 1.37),
                         fill_value='extrapolate',
                         bounds_error=False)(M)

        elif self.ifmr_model == 'manual':
            m = self.ifmr_function(M)

        else:

            raise ValueError('Please provide a valid model.')

        return m

    def _itp2D_gradient(self, f, val1, val2, frac=1e-6):
        '''
        A function to find the gradient in the direction in the first dimension
        of a 2D function at a given coordinate.

        Parameters
        ----------
        f: callable function
            A 2D function
        val1: float
            The first input value accepted by f. The gradient is computed in
            this direction.
        val2: float
            The first input value accepted by f.
        frac: float (Default: 1e-6)
            The small fractional increment of val1.

        Return
        ------
        Gradient in the direction of val1.

        '''

        if not callable(f):
            raise TypeError('f has to be a callable function.')

        increment = val1 * frac / 2.
        grad = np.asarray(
            (f(val1 + increment, val2) - f(val1 - increment, val2)) /
            (increment * 2.)).reshape(-1)

        # cooling((L+1), m) - cooling(L, m) must be negative
        grad[grad > 0.] = 0.
        grad[np.isnan(grad)] = 0.

        return grad

    def _find_M_min(self, M, logL, T0):
        '''
        A function to be minimised to find the minimum mass limit that a MS
        star could have turned into a WD in the given age, T0, of the
        population.

        Parameters
        ----------
        M: float
            MS mass.
        logL: float
            log WD luminosity.
        T0: float
            Age of the system.

        Return
        ------
        The difference between the total time and the sum of the cooling time
        and main sequence lifetime.

        '''

        # Get the WD mass
        m = self._ifmr(M)

        # Get the cooling age from the WD mass and the luminosity
        t_cool = self.cooling_interpolator(logL, m)
        if t_cool <= 0.:
            return np.inf

        # Get the MS life time
        t_ms = self._ms_age(M)
        if t_ms <= 0.:
            return np.inf

        t_diff = T0 - t_cool - t_ms

        if t_diff < 0.:

            return np.inf

        else:

            return M**2.

    def _integrand(self, M, logL, T0):
        '''
        The integrand of the number density computation based on the
        pre-selected (1) MS lifetime model, (2) initial mass function,
        (3) initial-final mass relation, and (4) WD cooling model.

        Parameters
        ----------
        M: float
            Main sequence stellar mass
        logL: float
            log(Luminosity) / (erg/s)
        T0: float
            Look-back time

        Return
        ------
        The product for integrating to the number density.

        '''
        # Get the WD mass
        m = self._ifmr(M)

        # Get the mass function
        MF = self._imf(M)

        # Get the WD cooling time
        t_cool = self.cooling_interpolator(logL, m)

        # Get the MS lifetime
        t_ms = self._ms_age(M)

        # Get the time since star formation
        time = T0 - t_cool - t_ms

        if time < 0.:

            return 0.

        # Get the SFR
        sfr = self.sfr(time)

        # Get the cooling rate
        dLdt = self.cooling_rate_interpolator(logL, m)

        if np.isfinite(dLdt) & np.isfinite(sfr):

            return MF * sfr * dLdt

        else:

            return 0.

    def set_sfr_model(self,
                      mode='constant',
                      duration=1E9,
                      decay_constant=6.9314718056e-10,
                      sfr_model=None):
        '''
        Set the SFR scenario, we only provide a few basic forms, free format
        can be supplied as a callable function.

        The SFR function accepts the time in unit of year, it is the time since
        T0 provided in compute_density().

        Parameters
        ----------
        mode: str (Default: 'constant')
            Choose from 'constant', 'burst', 'decay' and 'manual'
        duration: float (Default: 1E9)
            Only used if mode is 'burst'.
        decay_constant: float (Default: 6.9314718056e-10)
            Only used if mode is 'decay'. The default value has a SFR decay
            half-life of 1 Gyr.
        sfr_model: callable function (Default: None)
            The star formation rate at a given time since T0. In unit of years.
            If not callable, it uses a constant star formation rate.

        '''

        if mode == 'constant':

            self.sfr = None

        elif mode == 'burst':

            t0 = 0.
            t1 = duration
            t2 = 3e10

            self.sfr = interp1d(np.array(
                (-t2, t0 - 1e-10, t0, t1, t1 + 1e-10, t2)),
                                np.array((0., 0., 1., 1., 0., 0.)),
                                fill_value='extrapolate')

        elif mode == 'decay':

            t = 10.**np.linspace(0, 10.5, 10000)
            sfr = np.exp(-decay_constant * t)

            self.sfr = interp1d(t, sfr, fill_value='extrapolate')

        elif mode == 'manual':

            if callable(sfr_model):

                self.sfr = sfr_model

            else:

                warnings.warn('The sfr_model provided is not callable, '
                              'None is applied, i.e. constant star fomration.')
                self.sfr = None

        else:

            "Please choose a valid mode of SFR model."

        self.sfr_mode = mode

    def set_imf_model(self, model, ifmr_function=None):
        '''
        Set the main sequence total lifetime model.

        Parameter
        ---------
        model: str (Default: 'C16')
            Choice of IFMR model:
                1. K01 - Kroupa 2001
                2. C03 - Charbrier 2003
                3. C03b - Charbrier 2003 (including binary)
                4. manual
        imf_function: callable function (Default: None)
            A callable imf function, only used if model is 'manual'.

        '''

        self.imf_model = model
        self.imf_function = ifmr_function

    def set_ms_model(self, model, ms_function=None):
        '''
        Set the main sequence total lifetime model.

        Parameter
        ---------
        model: str (Default: 'C16')
            Choice of IFMR model:
                1. C16 - Choi et al. 2016
                2. Bressan - BASTI
                3. manual
        ifmr_function: callable function (Default: None)
            A callable ifmr function, only used if model is 'manual'.

        '''

        self.ms_model = model
        self.ms_function = ms_function

    def set_ifmr_model(self, model, ifmr_function=None):
        '''
        Set the initial-final mass relation (IFMR).

        Parameter
        ---------
        model: str (Default: 'EB18')
            Choice of IFMR model:
                1. C08 - Catalan et al. 2008
                2. C08b - Catalan et al. 2008 (two-part)
                3. S09 - Salaris et al. 2009
                4. S09b - Salaris et al. 2009 (two-part)
                5. W09 - Williams, Bolte & Koester 2009
                6. K09 - Kalirai et al. 2009
                7. K09b - Kalirai et al. 2009 (two-part)
                8. C18 - Cummings et al. 2018
                9. EB18 - El-Badry et al. 2018
                10. manual
        ifmr_function: callable function (Default: None)
            A callable ifmr function, only used if model is 'manual'.

        '''

        self.ifmr_model = model
        self.ifmr_function = ifmr_function

    def set_low_mass_cooling_model(self, model):
        '''
        Set the WD cooling model.

        Parameter
        ---------
        model: str (Default: 'montreal_co_da_20')
            Choice of WD cooling model:
            1. 'montreal_co_da_20' - Bedard et al. 2020 CO DA
            2. 'montreal_co_db_20' - Bedard et al. 2020 CO DB
            3. 'lpcode_he_da_07' - Panei et al. 2007 He DA
            4. 'lpcode_co_da_07' - Panei et al. 2007 CO DA
            5. 'lpcode_he_da_09' - Althaus et al. 2009 He DA

            The naming convention follows this format:
            [model]_[core composition]_[atmosphere]_[publication year]
            where a few models continue to have extra property description
            terms trailing after the year, currently they are either the
            progenitor metallicity or the (lack of) phase separation in the
            evolution model.

        '''

        if model in [
                'montreal_co_da_20', 'montreal_co_db_20', 'lpcode_he_da_07',
                'lpcode_co_da_07', 'lpcode_he_da_09', None
        ]:
            self.low_mass_cooling_model = model
        else:
            raise ValueError('Please provide a valid model.')

    def set_intermediate_mass_cooling_model(self, model):
        '''
        Set the WD cooling model.

        Parameter
        ---------
        model: str (Default: 'montreal_co_da_20')
            Choice of WD cooling model:
            1. 'montreal_co_da_20' - Bedard et al. 2020 CO DA
            2. 'montreal_co_db_20' - Bedard et al. 2020 CO DB
            3. 'lpcode_co_da_10_z001' - Renedo et al. 2010 CO DA Z=0.01
            4. 'lpcode_co_da_10_z0001' - Renedo et al. 2010 CO DA Z=0.001
            5. 'lpcode_co_da_15_z00003' - Althaus et al. 2015 DA Z=0.00003
            6. 'lpcode_co_da_15_z0001' - Althaus et al. 2015 DA Z=0.0001
            7. 'lpcode_co_da_15_z0005' - Althaus et al. 2015 DA Z=0.0005
            8. 'lpcode_co_da_17_y04' - Althaus et al. 2017 DB Y=0.4
            9. 'lpcode_co_db_17' - Camisassa et al. 2017 DB
            10. 'basti_co_da_10' - Salari et al. 2010 CO DA
            11. 'basti_co_db_10' - Salari et al. 2010 CO DB
            12. 'basti_co_da_10_nps' - Salari et al. 2010 CO DA,
                                       no phase separation
            13. 'basti_co_db_10_nps' - Salari et al. 2010 CO DB,
                                       no phase separation

            The naming convention follows this format:
            [model]_[core composition]_[atmosphere]_[publication year]
            where a few models continue to have extra property description
            terms trailing after the year, currently they are either the
            progenitor metallicity or the (lack of) phase separation in the
            evolution model.

        '''

        if model in [
                'montreal_co_da_20', 'montreal_co_db_20',
                'lpcode_co_da_10_z001', 'lpcode_co_da_10_z0001',
                'lpcode_co_da_15_z00003', 'lpcode_co_da_15_z0001',
                'lpcode_co_da_15_z0005', 'lpcode_co_da_17_y04',
                'lpcode_co_db_17', 'basti_co_da_10', 'basti_co_db_10',
                'basti_co_da_10_nps', 'basti_co_db_10_nps', None
        ]:
            self.intermediate_mass_cooling_model = model
        else:
            raise ValueError('Please provide a valid model.')

    def set_high_mass_cooling_model(self, model):
        '''
        Set the WD cooling model.

        Parameter
        ---------
        model: str (Default: 'montreal_co_da_20')
            Choice of WD cooling model:
            1. 'montreal_co_da_20' - Bedard et al. 2020 CO DA
            2. 'montreal_co_db_20' - Bedard et al. 2020 CO DB
            3. 'lpcode_one_da_07' - Althaus et al. 2007 ONe DA
            4. 'lpcode_one_da_19' - Camisassa et al. 2019 ONe DA
            5. 'lpcode_one_db_19' - Camisassa et al. 2019 ONe DB
            6. 'basti_co_da_10' - Salari et al. 2010 CO DA
            7. 'basti_co_db_10' - Salari et al. 2010 CO DB
            8. 'basti_co_da_10_nps' - Salari et al. 2010 CO DA,
                                      no phase separation
            9. 'basti_co_db_10_nps' - Salari et al. 2010 CO DB,
                                      no phase separation
            10. 'mesa_one_da_18' - Lauffer et al. 2018 ONe DA
            11. 'mesa_one_db_18' - Lauffer et al. 2018 ONe DB

            The naming convention follows this format:
            [model]_[core composition]_[atmosphere]_[publication year]
            where a few models continue to have extra property description
            terms trailing after the year, currently they are either the
            progenitor metallicity or the (lack of) phase separation in the
            evolution model.

        '''

        if model in [
                'montreal_co_da_20', 'montreal_co_db_20', 'lpcode_one_da_07',
                'lpcode_one_da_19', 'lpcode_one_db_19', 'basti_co_da_10',
                'basti_co_db_10', 'basti_co_da_10_nps', 'basti_co_db_10_nps',
                'mesa_one_da_18', 'mesa_one_db_18', None
        ]:
            self.high_mass_cooling_model = model
        else:
            raise ValueError('Please provide a valid model.')

    def compute_cooling_age_interpolator(self):
        '''
        Compute the callable CloughTocher2DInterpolator of the cooling time
        of WDs. It needs to use float64 or it runs into float-point error
        at very faint lumnosity.

        '''

        # Set the low mass cooling model, i.e. M < 0.5 M_sun
        if self.low_mass_cooling_model in [
                'montreal_co_da_20', 'montreal_co_db_20'
        ]:

            mass_low, cooling_model_low = bedard20_formatter(
                self.low_mass_cooling_model, mass_range='low')

        elif self.low_mass_cooling_model in [
                'lpcode_he_da_07', 'lpcode_co_da_07'
        ]:

            mass_low, cooling_model_low = panei07_formatter(
                self.low_mass_cooling_model)

        elif self.low_mass_cooling_model == 'lpcode_he_da_09':

            mass_low, cooling_model_low = althaus09_formatter(mass_range='low')

        elif self.low_mass_cooling_model == 'lpcode_co_da_17_y04':

            mass_low, cooling_model_low = althaus17_formatter(mass_range='low')

        elif self.low_mass_cooling_model is None:

            mass_low = None
            cooling_model_low = None

        else:

            raise ValueError('Invalid low mass model.')

        # Set the intermediate mass cooling model, i.e. 0.5 < M < 1.0 M_sun
        if self.intermediate_mass_cooling_model in [
                'montreal_co_da_20', 'montreal_co_db_20'
        ]:

            mass_intermediate, cooling_model_intermediate =\
                bedard20_formatter(
                    self.intermediate_mass_cooling_model,
                    mass_range='intermediate')

        elif self.intermediate_mass_cooling_model in [
                'lpcode_co_da_10_z001', 'lpcode_co_da_10_z0001'
        ]:

            mass_intermediate, cooling_model_intermediate =\
                renedo10_formatter(self.intermediate_mass_cooling_model)

        elif self.intermediate_mass_cooling_model in [
                'lpcode_co_da_15_z00003', 'lpcode_co_da_15_z0001',
                'lpcode_co_da_15_z0005'
        ]:

            mass_intermediate, cooling_model_intermediate =\
                althaus15_formatter(self.intermediate_mass_cooling_model)

        elif self.intermediate_mass_cooling_model == 'lpcode_co_da_17_y04':

            mass_intermediate, cooling_model_intermediate =\
                althaus17_formatter(mass_range='intermediate')

        elif self.intermediate_mass_cooling_model == 'lpcode_co_db_17':

            mass_intermediate, cooling_model_intermediate =\
                 camisassa17_formatter()

        elif self.intermediate_mass_cooling_model in [
                'basti_co_da_10', 'basti_co_db_10', 'basti_co_da_10_nps',
                'basti_co_db_10_nps'
        ]:

            mass_intermediate, cooling_model_intermediate =\
                salaris10_formatter(
                    self.intermediate_mass_cooling_model,
                    mass_range='intermediate')

        elif self.intermediate_mass_cooling_model is None:

            mass_intermediate = None
            cooling_model_intermediate = None

        else:

            raise ValueError('Invalid intermediate mass model.')

        # Set the high mass cooling model, i.e. 1.0 < M M_sun
        if self.high_mass_cooling_model in [
                'montreal_co_da_20', 'montreal_co_db_20'
        ]:

            mass_high, cooling_model_high = bedard20_formatter(
                self.high_mass_cooling_model, mass_range='high')

        elif self.high_mass_cooling_model == 'lpcode_one_da_07':

            mass_high, cooling_model_high = althaus07_formatter()

        elif self.high_mass_cooling_model in [
                'lpcode_one_da_19', 'lpcode_one_db_19'
        ]:

            mass_high, cooling_model_high = camisassa19_formatter(
                self.high_mass_cooling_model)

        elif self.high_mass_cooling_model in [
                'mesa_one_da_18', 'mesa_one_db_18'
        ]:

            mass_high, cooling_model_high = lauffer18_formatter(
                self.high_mass_cooling_model)

        elif self.high_mass_cooling_model in [
                'basti_co_da_10', 'basti_co_db_10', 'basti_co_da_10_nps',
                'basti_co_db_10_nps'
        ]:

            mass_high, cooling_model_high = salaris10_formatter(
                self.high_mass_cooling_model, mass_range='high')

        elif self.high_mass_cooling_model is None:

            mass_high = None
            cooling_model_high = None

        else:

            raise ValueError('Invalid high mass model.')

        # Gather all the models in different mass ranges

        if mass_low is not None:
            # Reshaping the WD mass array to match the shape of the other two.
            mass_low = np.concatenate(
                np.array([[mass_low[i]] * len(model['age'])
                          for i, model in enumerate(cooling_model_low)],
                         dtype=object)).T.ravel().astype(np.float64)

            # The luminosity of the WD at the corresponding mass and age
            luminosity_low = np.concatenate([
                i['lum'] for i in cooling_model_low
            ]).reshape(-1).astype(np.float64)

            # The luminosity of the WD at the corresponding mass and luminosity
            age_low = np.concatenate([i['age'] for i in cooling_model_low
                                      ]).reshape(-1).astype(np.float64)
        else:
            mass_low = []
            luminosity_low = []
            age_low = []
            cooling_model_low = []

        if mass_intermediate is not None:
            # Reshaping the WD mass array to match the shape of the other two.
            mass_intermediate = np.concatenate(
                np.array(
                    [[mass_intermediate[i]] * len(model['age'])
                     for i, model in enumerate(cooling_model_intermediate)],
                    dtype=object)).T.ravel().astype(np.float64)

            # The luminosity of the WD at the corresponding mass and age
            luminosity_intermediate = np.concatenate([
                i['lum'] for i in cooling_model_intermediate
            ]).reshape(-1).astype(np.float64)

            # The luminosity of the WD at the corresponding mass and luminosity
            age_intermediate = np.concatenate([
                i['age'] for i in cooling_model_intermediate
            ]).reshape(-1).astype(np.float64)
        else:
            mass_intermediate = []
            luminosity_intermediate = []
            age_intermediate = []
            cooling_model_intermediate = []

        if mass_high is not None:
            # Reshaping the WD mass array to match the shape of the other two.
            mass_high = np.concatenate(
                np.array([[mass_high[i]] * len(model['age'])
                          for i, model in enumerate(cooling_model_high)],
                         dtype=object)).T.ravel().astype(np.float64)

            # The luminosity of the WD at the corresponding mass and age
            luminosity_high = np.concatenate([
                i['lum'] for i in cooling_model_high
            ]).reshape(-1).astype(np.float64)

            # The luminosity of the WD at the corresponding mass and luminosity
            age_high = np.concatenate([i['age'] for i in cooling_model_high
                                       ]).reshape(-1).astype(np.float64)

        else:
            mass_high = []
            luminosity_high = []
            age_high = []
            cooling_model_high = []

        self.cooling_model_grid = np.concatenate(
            (cooling_model_low, cooling_model_intermediate,
             cooling_model_high))

        self.mass = np.concatenate((mass_low, mass_intermediate, mass_high))
        self.luminosity = np.concatenate(
            (luminosity_low, luminosity_intermediate, luminosity_high))
        self.age = np.concatenate((age_low, age_intermediate, age_high))

        self.cooling_interpolator = CloughTocher2DInterpolator(
            (np.log10(self.luminosity), self.mass),
            self.age,
            fill_value=-np.inf,
            tol=1e-10,
            maxiter=1000000,
            rescale=True)

        self.dLdt = -self._itp2D_gradient(self.cooling_interpolator,
                                          np.log10(self.luminosity), self.mass)

        finite_mask = np.isfinite(self.dLdt)

        self.cooling_rate_interpolator = CloughTocher2DInterpolator(
            (np.log10(self.luminosity)[finite_mask], self.mass[finite_mask]),
            self.dLdt[finite_mask],
            fill_value=0.,
            tol=1e-10,
            maxiter=1000000,
            rescale=True)

    def compute_density(self,
                        logL,
                        T0,
                        M_max=8.0,
                        limit=10000,
                        epsabs=1e-6,
                        epsrel=1e-6,
                        normed=True,
                        save_csv=False):
        '''
        Compute the density based on the pre-selected models: (1) MS lifetime
        model, (2) initial mass function, (3) initial-final mass relation, and
        (4) WD cooling model. It integrates over the function _integrand().

        Parameters
        ----------
        logL: float or array of float
            log(Luminosity) / (erg/s)
        T0: float or array of float
            Look-back time
        M_max: float (Deafult: 8.0)
            The upper limit of the main sequence stellar mass. This may not
            be used if it exceeds the upper bound of the IFMR model.
        limit: int (Default: 1000000)
            The maximum number of steps of integration
        epsabs: float (Default: 1e-6)
            The absolute tolerance of the integration step
        epsrel: float (Default: 1e-6)
            The relative tolerance of the integration step
        normed: boolean (Default: True)
            Set to True to return a WDLF sum to 1. Otherwise, it is arbitrary
            to the integrator.
        save_csv: boolean (Default: False)
            Set to True to save the WDLF as CSV files. One CSV per T0.

        '''

        if self.cooling_interpolator is None:

            self.compute_cooling_age_interpolator()

        T0 = np.asarray(T0).reshape(-1)
        logL = np.asarray(logL).reshape(-1)

        number_density = np.zeros((len(T0), len(logL)))

        for j, t in enumerate(T0):

            print('Computing T0 = {0:.2f} Gyr.'.format(t/1e9))

            M_upper_bound = M_max

            for i, logL_i in enumerate(logL):

                M_min = optimize.fminbound(self._find_M_min,
                                           0.5,
                                           M_upper_bound,
                                           args=(logL_i, t),
                                           xtol=1e-5,
                                           maxfun=10000)

                number_density[j][i] = integrate.quad(self._integrand,
                                                      M_min,
                                                      M_max,
                                                      args=(logL_i, t),
                                                      limit=limit,
                                                      epsabs=epsabs,
                                                      epsrel=epsrel)[0]

                M_upper_bound = M_min

            # Normalise the WDLF
            if normed:
                number_density[j] /= np.sum(number_density[j])

            if save_csv:

                filename = "{0:.2f}".format(t/1e9) + 'Gyr_' + self.sfr_mode + '_' +\
                    self.ms_model + '_' + self.ifmr_model + '_' +\
                    self.low_mass_cooling_model + '_' +\
                    self.intermediate_mass_cooling_model + '_' +\
                    self.high_mass_cooling_model + '.csv'
                np.savetxt(filename,
                           np.column_stack((logL, number_density[j])),
                           delimiter=',')

        self.logL = logL
        self.T0 = T0
        self.number_density = number_density

        return logL, number_density

    def plot_cooling_model(self,
                           mag=True,
                           display=True,
                           savefig=False,
                           filename=None):

        plt.figure(figsize=(12, 8))

        if mag:
            plt.scatter(np.log10(self.age),
                        -2.5 * np.log10(self.luminosity / 3.826E33) + 4.75,
                        c=self.mass,
                        s=5)
            plt.ylabel(r'M$_{\mathrm{bol}}$ / mag')
        else:
            plt.scatter(np.log10(self.age),
                        np.log10(self.luminosity / 3.826E33),
                        c=self.mass,
                        s=5)
            plt.ylabel(r'L/L$_{\odot}$')

        plt.xlim(6, 10.5)
        plt.xlabel(r'Age / Gyr')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Solar Mass', rotation=270)
        plt.grid()
        plt.legend()
        plt.tight_layout()

        if savefig:
            if filename is None:
                filename = self.low_mass_cooling_model + '_' +\
                    self.intermediate_mass_cooling_model + '_' +\
                    self.high_mass_cooling_model + '.png'
            plt.savefig(filename)

        if display:
            plt.show()

    def plot_wdlf(self, mag=True, display=True, savefig=False, filename=None):

        plt.figure(figsize=(12, 8))

        for i, t in enumerate(self.T0):

            if mag:

                plt.plot(-2.5 * (self.logL - np.log10(3.826E33)) + 4.75,
                         np.log10(self.number_density[i]),
                         label="{0:.2f}".format(t/1e9) + ' Gyr')
                plt.xlim(4, 20)
                plt.xlabel(r'M$_{\mathrm{bol}}$ / mag')

            else:

                plt.plot(self.logL - np.log10(3.826E33),
                         np.log10(self.number_density[i]),
                         label="{0:.2f}".format(t/1e9) + ' Gyr')
                plt.xlim(0, -5)
                plt.xlabel(r'L/L$_{\odot}$')

        plt.ylabel(r'$\log{(N)}$')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        if savefig:
            if filename is None:
                filename = self.sfr_mode + '_' + self.ms_model + '_' +\
                    self.ifmr_model + '_' + self.low_mass_cooling_model +\
                    '_' + self.intermediate_mass_cooling_model + '_' +\
                    self.high_mass_cooling_model + '.png'
            plt.savefig(filename)

        if display:
            plt.show()
