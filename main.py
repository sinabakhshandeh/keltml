from kelt_ml import Kelt
import matplotlib.pyplot as plt

t0 = 2456924.6845
path = 'KELT_N04_lc_040968_V01_comb_raw.dat'
nterms = 5
min_freq = 0.05
max_freq = 0.2

kelt = Kelt(path, t0)
kelt.lomb_scargel(nterms, min_freq, max_freq)
kelt.binned_statistic()