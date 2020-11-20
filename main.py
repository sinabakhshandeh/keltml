from kelt_ml import Kelt
import matplotlib.pyplot as plt

t0 = 2456924.6845
path = 'KELT_N04_lc_040968_V01_comb_raw.dat'
min_p = 1
max_p = 30

kelt = Kelt(path)
kelt.period_finder(method = 'BLS_astropy')
kelt.binned_statistic()

print('done')