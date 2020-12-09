import numpy as np
from matplotlib import pyplot as plt
from WDLFBuilder.atmosphere_model_reader import interp_atm

# Default passband is G3
G = interp_atm()
BP = interp_atm(passband='G3_BP')
RP = interp_atm(passband='G3_RP')

logg = np.arange(7., 10., 0.5)
Mbol = np.arange(0., 20., 0.1)

plt.figure(1, figsize=(8, 8))
for logg_i in logg:
    plt.plot(BP(logg_i, Mbol) - RP(logg_i, Mbol),
             G(logg_i, Mbol),
             label="$\log(g) = {}$".format(logg_i))

plt.ylim(20., 6.)
plt.grid()
plt.legend()
plt.xlabel('(BP - RP) / mag')
plt.ylabel('G / mag')
plt.title('DA Cooling tracks')
plt.tight_layout()
plt.savefig('DA_cooling_tracks.png')
