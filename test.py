import numpy as np
from matplotlib import pyplot as plt

from theoretical_lf import WDLF

wdlf = WDLF()

#wdlf.set_imf_model('K01')
wdlf.compute_cooling_age_interpolator()


L = 10.**np.arange(27., 34., 0.05)
age = 1E9 * np.arange(6, 12, 1)
num = np.zeros((len(age), len(L)))

for i, t in enumerate(age):
    num[i] = wdlf.compute_density(L=L, T0=t)

plt.figure(1, figsize=(12, 8))

for i, t in enumerate(age):
    plt.plot(np.log10(L / 3.826E33),
             np.log10(num[i]),
             label=str(t * 1e-9) + ' Gyr')
    plt.xlabel(r'$\log{L / L_{\odot}}$')
    plt.ylabel(r'$\log{(N)}$')

plt.grid()
plt.legend()
plt.tight_layout()
plt.gca().invert_xaxis()

plt.figure(2, figsize=(12, 8))

for i, t in enumerate(age):
    plt.plot(-2.5 * np.log10(L / 3.826E33) + 4.75,
             np.log10(num[i]),
             label=str(t * 1e-9) + ' Gyr')
    plt.xlabel(r'M$_{\mathrm{bol}}$ / mag')
    plt.ylabel(r'$\log{(N)}$')

plt.grid()
plt.legend()
plt.tight_layout()

plt.show()
