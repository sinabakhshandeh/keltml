import numpy as np
import pandas as pd
from astropy.stats import LombScargle
import matplotlib.pyplot as plt
import scipy.stats as st


class Kelt:
    
    KELT_BINS = 250

    def __init__(self, path, t0) -> None:
        self.path = path
        self.t0 = t0
        self.data = np.loadtxt(path)
        self.df = pd.DataFrame({'t': self.data[:, 0], 'm': self.data[:, 1], 'e': self.data[:, 2]})
        self.df['m_median'] = self.df.m - np.median(self.df.m)
        self.julian_day_fixer()
        self.plot()

    def plot(self):
        plt.plot(self.df.t, self.df.m, 'b.')
        plt.gca().invert_yaxis()
        plt.show()
        plt.plot(self.df.t, self.df.m_median, 'b.')
        plt.gca().invert_yaxis()
        plt.show()

    def plot_ls(self):
        # fig, ax = plt.subplots()
        # ax.plot(1/self.frequency, self.power)
        # ax.set_ylabel("Power")
        # ax.set_xlabel("Period (d)")
        # ax.set_xlim(5,20)
        # # fig.set_size_inches(12.0,10.0)

        plt.plot(1 / self.frequency, self.power)
        # plt.set_ylabel("Power")
        # plt.set_xlabel("Period (d)")
        # plt.set_xlim(5,20)
        plt.show()

    def julian_day_fixer(self):
        if self.df.t[0] > 10000:
            self.df.t_kelt = self.df.t - 2450000
        else:
            self.df.t_kelt = self.df.t

    def t_0_fixer(self):
        self.phase0 = (self.t0 / self.period) - np.fix(self.t0 / self.period)
        self.phase = self.phase - self.phase0 - 0.25
        for i in np.arange(len(self.phase)):
            if self.phase[i] < 0:
                self.phase[i] += 1
            if self.phase[i] > 1:
                self.phase[i] -= 1

    def lomb_scargel(self, nt, min_f, max_f):
        ls = LombScargle(self.df.t.values, self.df.m_median.values, self.df.e.values, nterms=nt)
        self.frequency, self.power = ls.autopower(minimum_frequency=min_f, maximum_frequency=max_f)
        self.plot_ls()
        self.set_period()

    def set_period(self):
        self.period = 2 * (1. / self.frequency[np.argmax(self.power)])
        self.set_phase()

    def set_phase(self):
        self.phase = (self.df.t_kelt / self.period) - np.fix(self.df.t_kelt / self.period)
        self.t_0_fixer()

    def plot_binned(self):
        plt.plot(self.phase, self.df.m_median, 'y.', markersize=5)
        plt.plot(self.bin_middles_kelt, self.med_stat_kelt, 'k', markersize=6)
        plt.axis([0, 1, self.max_m, self.min_m])
        plt.show()

    def binned_statistic(self):
        self.max_m = ((30 * (max(self.df.m_median) - min(self.df.m_median))) / 100) + min(self.df.m_median)
        self.min_m = ((10 * (max(self.df.m_median) - min(self.df.m_median))) / 100) + min(self.df.m_median)

        (self.med_stat_kelt, bin_edges, binnumber) = st.binned_statistic(self.phase, self.df.m_median,
                                                                         statistic='median', bins=Kelt.KELT_BINS)
        r = (bin_edges[1] - bin_edges[0]) / 2

        self.bin_middles_kelt = [bin_edges[index] + r for index in np.arange(len(bin_edges) - 1)]
        self.plot_binned()
