from kltml import Kelt

path = 'KELT_N04_lc_040968_V01_comb_raw.dat'

kelt = Kelt(path)
# kelt.set_power_period(method='BLS_astropy')
kelt.set_power_period(method='LS_astropy')

kelt.weighted_mean_flux()
kelt.normalized_flux_std()
kelt.normalized_amplitude()
kelt.normalized_MAD()
kelt.beyond_1std()
kelt.skew()


print(f"Weigh:{kelt.mean}")
print(f"STD:{kelt.std}")
print(f"AMP:{kelt.amp}")
print(f"MAD:{kelt.mad}")
print(f"Beyon:{kelt.beyond}")
print(f"skew:{kelt.skew}")
