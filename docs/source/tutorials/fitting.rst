Photometric Fitting 
===================

Photometric fitting with the Montreal DA & DB models is possible with 2 minimisers and a MCMC sampling method. The fitter can solve for the effective temperature/bolometric magnitude (always fitted for), surface gravity (if not provided), and distance (if not provided). It also handles interstellar reddening, though it does not solve for the reddening but it merely takes into account the reddening in the fitting procedure. Below is a table summarising the possible combination of inputs to the fitter. For obvious reasons, the photometry of a target has to be provided in order to get the photometry fitted. A :math:`\checkmark` refers to when the parameter indicated by the column header is provided.

+------+--------------------+--------------------+--------------------+--------------------+
| Case | Photometry         | log(g)             | Distance           | Reddening          |
+------+--------------------+--------------------+--------------------+--------------------+
| 1    + :math:`\checkmark` | :math:`\checkmark` | :math:`\checkmark` | :math:`\checkmark` |
+------+--------------------+--------------------+--------------------+--------------------+
| 2    + :math:`\checkmark` |                    | :math:`\checkmark` | :math:`\checkmark` |
+------+--------------------+--------------------+--------------------+--------------------+
| 3    + :math:`\checkmark` | :math:`\checkmark` |                    | :math:`\checkmark` |
+------+--------------------+--------------------+--------------------+--------------------+
| 4    + :math:`\checkmark` | :math:`\checkmark` | :math:`\checkmark` |                    |
+------+--------------------+--------------------+--------------------+--------------------+
| 5    + :math:`\checkmark` | :math:`\checkmark` |                    |                    |
+------+--------------------+--------------------+--------------------+--------------------+
| 6    + :math:`\checkmark` |                    | :math:`\checkmark` |                    |
+------+--------------------+--------------------+--------------------+--------------------+
| 7    + :math:`\checkmark` |                    |                    | :math:`\checkmark` |
+------+--------------------+--------------------+--------------------+--------------------+
| 8    + :math:`\checkmark` |                    |                    |                    |
+------+--------------------+--------------------+--------------------+--------------------+

Below is the default setup of the `fit` method, we will go through the 8 cases one by one and see how we can configure the fitter to perform fitting in each scenario.

.. code:: python

    def fit(
        self,
        atmosphere=["H", "He"],
        filters=["G3", "G3_BP", "G3_RP"],
        mags=[None, None, None],
        mag_errors=[1.0, 1.0, 1.0],
        allow_none=False,
        distance=None,
        distance_err=None,
        extinction_convolved=True,
        kind="cubic",
        Rv=0.0,
        ebv=0.0,
        independent=["Mbol", "logg"],
        initial_guess=[10.0, 8.0],
        logg=8.0,
        atmosphere_interpolator="RBF",
        reuse_interpolator=False,
        method="minimize",
        nwalkers=100,
        nsteps=1000,
        nburns=100,
        progress=True,
        refine=False,
        refine_bounds=[5.0, 95.0],
        kwargs_for_RBF={},
        kwargs_for_CT={},
        kwargs_for_minimize={},
        kwargs_for_least_squares={},
        kwargs_for_emcee={},
    ):

    ...
    ...
    ...

Below shows the example of how the fitter can be configured to fit differently, the argument that is worth particular mentioning is the `independent`. It refers to the mathemtical term *independent variable*, which are the parameters to be fitted. It accepts a list of 1 or 2 strings, one from `["Mbol", "Teff"]`, and one from `["logg", "mass"]`. When distance is to be fitted, it is not configured here, but instead a `None` should be provided to the `distance` argument. See further down for the examples. When a distance is provided, **its uncertainty has to be provided**.

**Case 1**

When a log(g) is not included in the `independent` list, it will assume a fixed surface gravitiy as provided by `logg`, which is defaulted to 8.0, in this case we want to fit for the bolometric magnitude with a surface gravity of 7.81 for a DA at 21.0 pc with a reddening of `E(B-V) = 0.1` and `Rv` of 3.1 where the extinction is computed by convolving the filter profiles with the DA spectra. The magnitudes and uncertainties in the Gaia eDR3 are some variables `a`, `b`, and `c`.

.. code:: python

    from WDPhotTools.fitter import WDfitter
    ftr = WDfitter()

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=21.0,
            distance_err=0.1,
            extinction_convolved=True,
            kind="cubic",
            Rv=3.1,
            ebv=0.1,
            independent=["Mbol"],
            initial_guess=[10.0],
            logg=7.81)

**Case 2**

Compared to case 1, this fits for the logg, so we needs to add `"logg"` to the `independent` list, when that is in the list, the `logg` provided to the function will be **ignored** (i.e. whether is is 0.0, 7.81, 8.0 or Nan, it does not matter).

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=21.0,
            distance_err=0.1,
            extinction_convolved=True,
            kind="cubic",
            Rv=3.1,
            ebv=0.1,
            independent=["Mbol", "logg"],
            initial_guess=[10.0],
            logg=7.81)

**Case 3**

Compared to case 1, this fits for the distance, but we need to change two things, first is to set `distnace` to None, second is to provide a second value in the `initial_guess`, say, 30.7 pc (whenever distance is to be fitted, it should be appended to the end of the `initial_guess`).

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=None,
            extinction_convolved=True,
            kind="cubic",
            Rv=3.1,
            ebv=0.1,
            independent=["Mbol"],
            initial_guess=[10.0, 30.7],
            logg=7.81)

**Case 4**

This requires a very simple change, compared to case 1, we change `ebv` to 0.0, `Rv` will get ignored.

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=21.0,
            distance_err=0.1,
            ebv=0.0,
            independent=["Mbol"],
            initial_guess=[10.0],
            logg=7.81)

**Case 5**

This is a combination of case 3 and 4, and on top, if we opt to use the other interpolator and the `scipy.minimize.least_squares` minimiser, we can modify the `fit` to

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=None,
            ebv=0.0,
            independent=["Mbol"],
            initial_guess=[10.0, 30.7],
            logg=7.81,
            atmosphere_interpolator='RBF',
            method="least_squares"
            )

**Case 6**

This is a combination of case 2 and 4. We are also demonstrating how to modify the setting of the RBF interpolator and the walker number and step size for the sampling with `emcee` (finer control can be performed by supplying a dictionary through `kwargs_for_emcee`). At the end of the emcee, the solution will also get refined with a `scipy.minimize.minimize` minimiser bounded within the central 95% of the posterior distribution.

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=20.1,
            distance_err=0.1,
            ebv=0.0,
            independent=["Mbol", "logg"],
            initial_guess=[10.0, 8.0],
            atmosphere_interpolator='RBF',
            method="emcee",
            nwalkers=50,
            nsteps=2000,
            nburns=200,
            refine=True,
            refine_bounds=[2.5, 97.5],
            kwargs_for_RBF={"kernel": quintic}
            )


**Case 7**

This is the setup that is the most likely to fail because it is fitting 3 unknowns (Mbol/Teff, mass/logg and distance) while applying interstellar reddening based on an independent variable (distance) at each step of the minimisation. Note that the `independent` argument is supplied with a list of size 2 and `distance` is set to `None`, while the `initial_guess` is supplying 3 starting values for the Mobl, logg and distance (whenever distance is to be fitted, it should be appended to the end of the `initial_guess`). We switch back to the default interpolator in this example, which is "CT", and we reduce the tolerance to 1e-12 (which is unnecessarily precise but just as an example). Use this fitting with caution...

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=None,
            Rv=3.1,
            ebv=0.1,
            independent=["Mbol", "logg"],
            initial_guess=[10.0, 8.0, 30.0],
            method="emcee",
            nwalkers=50,
            nsteps=2000,
            nburns=200,
            kwargs_for_CT={"tol": 1e-12}
            )

**Case 8**

This is the same as case 7 except the reddening is not considered (ebv is set to 0.0), this makes the fitting a bit more stable but whenever the distance is fitted, use with caution...

.. code:: python

    ftr.fit(atmosphere=["H"],
            filters=["G3", "G3_BP", "G3_RP"],
            mags=[a, b, c],
            mag_errors=[a_err, b_err, c_err],
            distance=None,
            ebv=0.0,
            independent=["Mbol", "logg"],
            initial_guess=[10.0, 8.0, 30.0],
            method="emcee",
            nwalkers=50,
            nsteps=2000,
            nburns=200,
            kwargs_for_CT={"tol": 1e-12}
            )

