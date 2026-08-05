=================================
Interpolator Extrapolation Limits
=================================

`AtmosphereModelReader.interp_am` and
`CoolingModelReader.compute_cooling_age_interpolator` keep
``allow_extrapolation=False`` by default.

When extrapolation is enabled:

- queries are limited to at most 50% beyond each axis span,
- non-finite outputs are replaced by smooth fallback interpolation,
- physically impossible outputs are sanitised:

  - atmosphere ``Teff/logg/mass/age`` are forced to finite positive values,
  - cooling age is forced to finite positive values,
  - cooling rate (``dL/dt``) is forced to finite non-positive values.

Estimated success rates below were measured with 1000 random samples at each
extrapolation fraction and report the fraction of *extrapolated* outputs that
remained physical and finite.

Atmosphere Interpolator (DA, ``dependent="Teff"``, ``independent=["logg","Mbol"]``)
--------------------------------------------------------------------------------------

+----------------------+---------+----------+
| Extrapolation Level  | CT      | RBF      |
+======================+=========+==========+
| 10%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 20%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 30%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 40%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 50%                  | 1.000   | 1.000    |
+----------------------+---------+----------+

Cooling Interpolator (age + cooling-rate physical checks)
----------------------------------------------------------

+----------------------+---------+----------+
| Extrapolation Level  | CT      | RBF      |
+======================+=========+==========+
| 10%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 20%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 30%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 40%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
| 50%                  | 1.000   | 1.000    |
+----------------------+---------+----------+
