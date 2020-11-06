from io import StringIO
import glob
import os
import numpy as np
import scipy
from scipy import optimize, integrate
from scipy.interpolate import interp1d
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.misc import derivative
import warnings


def itp2D_gradient(f, val1, val2, frac=1e-6):
    '''
    A function to find the gradient in the direction in the first dimension of
    a 2D function at a given coordinate.

    Parameters
    ----------
    f: callable function
        A 2D function
    val1: float
        The first input value accepted by f. The gradient is computed in this
        direction.
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
    grad = np.asarray((f(val1 + increment, val2) - f(val1 - increment, val2)) /
                      (increment * 2.)).reshape(-1)

    # cooling((L+1), m) - cooling(L, m) must be negative
    grad[grad > 0.] = 0.

    return grad


class WDLF:
    '''
    Computing the theoretical WDLFs based on the input IFMR, WD cooling and
    MS lifetime models.

    We are using little m for WD mass and big M for MS mass throughout this
    package.

    The unit for (1) mass is solar mass, (2) luminosity is erg/s,
    (3) time/age is in year.

    '''

    def __init__(self,
                 imf_model='C03',
                 ifmr_model='EB18',
                 cooling_model='montreal_thick',
                 ms_model='Bressan'):
        # The IFMR, WD cooling and MS lifetime models are required to
        # initialise the object.
        self.set_imf_model(imf_model)
        self.set_ifmr_model(ifmr_model)
        self.set_cooling_model(cooling_model)
        self.set_ms_model(ms_model)

        self.cooling_interpolator = None

    def _montreal_formatter(self, model):
        '''
        A formatter to load the Montreal WD cooling model from
        http://www.astro.umontreal.ca/~bergeron/CoolingModels/

        '''

        if model == 'montreal_thin':
            filelist = glob.glob('wd_cooling/montreal/*thin*')

        if model == 'montreal_thick':
            filelist = glob.glob('wd_cooling/montreal/*thick*')

        column_key = np.array(('step', 'Teff', 'logg', 'rayon', 'age', 'lum',
                               'logTc', 'logPc', 'logrhoc', 'MxM', 'logqx',
                               'lumnu', 'logH', 'logHe', 'logC', 'logO'))
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        mass = np.array([i.split('_')[1]
                         for i in filelist]).astype(np.float64) / 100.
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

            cooling_model[i] = np.loadtxt(StringIO(cooling_model_text),
                                          dtype=dtype)

        return mass, cooling_model

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

        elif self.ifmr_model == 'Manual':
            m = self.ifmr_function(M)

        else:

            raise ValueError('Please provide a valid model.')

        return m

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
                M_mask = M < 1.
                # 0.158 / (ln(10) * M) = 0.06861852814 / M
                # log(0.079) = -1.1023729087095586
                # 2 * 0.69**2. = 0.9522
                MF[M_mask] = (0.06861852814 / M[M_mask]) * np.exp(
                    -(np.log10(M[M_mask]) + 1.1023729087095586)**2. / 0.9522)
                # Normalise to m=1
                ratio = MF[max(np.where(M_mask)[0])]
                MF[M_mask] /= ratio

        elif self.imf_model == 'C03b':

            MF = M**-2.3

            if (M < 1).any():
                M_mask = M < 1.
                # 0.086 * 1. / (ln(10) * M) = 0.03734932544 / M
                # log(0.22) = -0.65757731917
                # 2 * 0.57**2. = 0.6498
                MF[M_mask] = (0.03734932544 / M[M_mask]) * np.exp(
                    -(np.log10(M[M_mask]) + 0.65757731917)**2. / 0.6498)
                # Normalise to M=1
                ratio = MF[max(np.where(M_mask)[0])]
                MF[M_mask] /= ratio

        elif self.imf_model == 'Manual':

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
            datatable = np.loadtxt('ms_lifetime/bressan00170279.csv', delimiter=',')
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

    def _integrand(self, M, L, T0, SFR=None):
        '''
        The integrand of the number density computation based on the
        pre-selected (1) MS lifetime model, (2) initial mass function,
        (3) initial-final mass relation, and (4) WD cooling model.

        Parameters
        ----------
        M: float
            Main sequence stellar mass
        L: float
            Luminosity
        T0: float
            Look-back time
        SFR: callable function (Default: None)
            The star formation rate at a given time since T0. In unit of years.
            If not callable, it uses a constant star formation rate.

        Return
        ------
        The product for integrating to the number density.

        '''

        if self.cooling_interpolator is None:
            self._compute_cooling_age_interpolator()

        # Get the WD mass
        m = self._ifmr(M)
        if m == M:
            return 0.

        # Get the mass function
        MF = self._imf(M)

        # Get the WD cooling time
        t_cool = self.cooling_interpolator(L, m)
        if t_cool < 0.:
            return 0.

        # Get the MS lifetime
        t_ms = self._ms_age(M)
        if t_ms < 0.:
            return 0.

        # Get the time since star formation
        time = T0 - t_cool - t_ms

        # Get the derivative of the cooling rate
        dLdt = -itp2D_gradient(self.cooling_interpolator, L, m)

        # If a callable function is NOT provided, return constant SRF
        if not callable(SFR):
            sfr = 1.
        else:
            sfr = SFR(time)

        if dLdt > 0.:

            return MF * sfr * dLdt

        else:

            return 0.

    def _Mmin(self, M, L, T0):
        '''
        A function to be minimised to find the minimum mass limit that a MS
        star could have turned into a WD in the given age, T0, of the 
        population.

        Parameters
        ----------
        M: float
            MS mass.
        L: float
            WD luminosity.
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
        t_cool = self.cooling_interpolator(L, m)

        # If the cooling time is longer than the total age, reject
        if t_cool > T0:

            return np.inf

        if np.isnan(t_cool) or (t_cool == 0):
            return np.inf

        # Get the MS life time
        t_ms = self._ms_age(M)

        # If the MS life time exceeds the total age, rejct
        if t_ms > T0:

            return np.inf

        # Get the difference between the total age and the cooling + MS time
        T_diff = T0 - t_cool - t_ms

        # If the value is not numeric, reject
        if (not np.isfinite(T_diff)) or (T_diff < 0):

            return np.inf

        # Else return the square of the time difference, we want it to be as
        # close to 0 as possible
        else:

            return T_diff**2.

    def compute_cooling_age_interpolator(self):
        '''
        Compute the callable CloughTocher2DInterpolator of the cooling time
        of WDs. It needs to use float64 or it runs into float-point error
        at very faint lumnosity.

        '''

        if self.cooling_model in ['montreal_thin', 'montreal_thick']:
            mass, cooling_model = self._montreal_formatter(self.cooling_model)

            # Reshaping the WD mass array to match the shape of the other two.
            mass = np.concatenate(
                np.array([[mass[i]] * len(model['age'])
                          for i, model in enumerate(cooling_model)],
                         dtype=object)).T.ravel().astype(np.float64)
            # The luminosity of the WD at the corresponding mass and age
            luminosity = np.concatenate([i['lum'] for i in cooling_model
                                         ]).reshape(-1).astype(np.float64)
            # The luminosity of the WD at the corresponding mass and luminosity
            age = np.concatenate([i['age'] for i in cooling_model
                                  ]).reshape(-1).astype(np.float64)

            self.mass = mass
            self.cooling_model_grid = cooling_model
            self.cooling_interpolator = CloughTocher2DInterpolator(
                (luminosity, mass),
                age,
                fill_value=0.,
                tol=1e-10,
                maxiter=1e6,
                rescale=True)

        else:

            raise ValueError(
                'Please choose from montreal_thin and montreal_thick. Use '
                'set_cooling_model() to change to a valid model.')

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
                10. Manual
        ifmr_function: callable function (Default: None)
            A callable ifmr function, only used if model is 'Manual'.

        '''

        self.ifmr_model = model
        self.ifmr_function = ifmr_function

    def set_cooling_model(self, model):
        '''
        Set the WD cooling model.

        Parameter
        ---------
        model: str (Default: 'montreal_thick')
            Choice of WD cooling model:
                1. 'montreal_thick' - Montreal hydrogen atmospheremodel (2020)
                2. 'montreal_thin' - Montreal helium atmosphere model (2020)
                3. 'laplata' - La Plata model (2000)
                4. 'basti' - BASTI model (2000)
                5. 'bastips' - BASTI model with phase separation (2000)

        '''

        self.cooling_model = model

    def set_ms_model(self, model, ms_function=None):
        '''
        Set the main sequence total lifetime model.

        Parameter
        ---------
        model: str (Default: 'C16')
            Choice of IFMR model:
                1. C16 - Choi et al. 2016
                2. Bressan - BASTI
                3. Manual
        ifmr_function: callable function (Default: None)
            A callable ifmr function, only used if model is 'Manual'.

        '''

        self.ms_model = model
        self.ms_function = ms_function

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
                4. Manual
        imf_function: callable function (Default: None)
            A callable imf function, only used if model is 'Manual'.

        '''

        self.imf_model = model
        self.imf_function = ifmr_function

    def compute_density(self,
                        L,
                        T0,
                        M_max=8.0,
                        SFR=None,
                        limit=1000000,
                        tolerance=1e-10):
        '''
        Compute the density based on the pre-selected models: (1) MS lifetime
        model, (2) initial mass function, (3) initial-final mass relation, and
        (4) WD cooling model. It integrates over the function _integrand().

        Parameters
        ----------
        L: float
            Luminosity
        T0: float
            Look-back time
        M_max: float (Deafult: 8.0)
            The upper limit of the main sequence stellar mass. This may not
            be used if it exceeds the upper bound of the IFMR model.
        SFR: callable function (Default: None)
            The star formation rate at a given time since T0. In unit of years.
            If not callable, it uses a constant star formation rate.
        limit: int (Default: 1000000)
            The maximum number of steps of integration
        tolerance: float (Default: 1e-10)
            The relative tolerance of the integration step

        Return
        ------
        The integrated number density.

        '''

        number_density = np.zeros_like(L)

        # In our computation, the mass limit is decreasing in each step, so we
        # include that as our bound to improve the rate of convergence
        M_bound = M_max

        for i, L_i in enumerate(L):

            M_min = optimize.fminbound(self._Mmin,
                                       0.6,
                                       M_bound,
                                       args=(L_i, T0),
                                       full_output=0,
                                       maxfun=1e5,
                                       disp=0)

            if (M_min < M_max) & (M_min > 0.6):
                number_density[i] = integrate.quad(self._integrand,
                                                   M_min,
                                                   M_max,
                                                   args=(L_i, T0, SFR),
                                                   limit=limit,
                                                   epsrel=tolerance)[0] * L_i
                M_bound = M_min

        return number_density
