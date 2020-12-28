import os
import sys

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from scipy import stats


class Kelt:

    def __init__(self, path, t0=0):
        self.path = path
        self.t0 = t0
        self.data = np.loadtxt(path)
        self.df = pd.DataFrame({'t': self.data[:, 0], 'm': self.data[:, 1], 'dflux': self.data[:, 2]})
        if self.df.t[0] > 10000:  # j_d_fixer
            self.df['t_k'] = self.df.t - 2450000
        else:
            self.df['t_k'] = self.df.t
        print('before flux')
        self.df['flux'] = 10 ** (-0.4 * (self.df.m - 1.08))

        print(self.__dict__.keys())

    def set_power_period(self, nt=5, min_p=1, max_p=100, n_f=10000, auto=True, method='LS_astropy'):
        self.pmin = min_p
        self.pmax = max_p
        self.method = method
        if self.method == 'LS_astropy':
            if auto:
                ls = LombScargle(self.df.t.values, self.df.m.values, self.df.dflux.values, nterms=nt)
                self.frequency, self.power = ls.autopower(minimum_frequency=1. / self.pmax,
                                                          maximum_frequency=1. / self.pmin)
            else:
                self.frequency = np.linspace(1. / self.pmax, 1. / self.pmin, n_f)
                self.power = LombScargle(self.df.t.values, self.df.m.values, self.df.dflux.values).power(self.frequency)

        elif self.method == 'BLS_astropy':
            model = BoxLeastSquares(self.df.t.values, self.df.m.values, dy=self.df.dflux.values)
            if auto:
                periodogram = model.autopower(0.2)
                self.frequency = 1. / periodogram.period
                self.power = periodogram.power
            else:
                periods = np.linspace(self.pmin, self.pmax, 10)
                periodogram = model.power(periods, 0.2)
                self.frequency = 1. / periodogram.period
                self.power = periodogram.power
        else:
            print('Method should be chosen between these options:')
            print('LS_astropy, BLS_astropy')
            sys.exit()
        # setting_period
        period = (1. / self.frequency[np.argmax(self.power)])
        print("p f p-f", period, np.fix(period), period-np.fix(period))
        if period - np.fix(period) < 0.009:
            self.period = (1. / self.frequency[(np.asarray(self.power).argsort()[-2])])
        else:
            self.period = period
        # print(f"Period: {self.period} Power:{self.power}")

    def set_phase(self):
        self.phase = (self.df.t_kelt / self.period) - np.fix(self.df.t_kelt / self.period)
        # t_0_fixer
        self.phase_0 = (self.t0 / self.period) - np.fix(self.t0 / self.period)
        self.phase -= self.phase_0 - 0.25
        for i in np.arange(len(self.phase)):
            if self.phase[i] < 0:
                self.phase[i] += 1
            if self.phase[i] > 1:
                self.phase[i] -= 1

    def weighted_mean_flux(self):
        """Measure (SNR weighted) mean flux in griz"""

        weighted_mean = lambda flux, dflux: np.sum(flux * (flux / dflux) ** 2) / np.sum((flux / dflux) ** 2)

        flux = getattr(self, "df").flux
        dflux = getattr(self, "df").dflux
        setattr(self, 'mean', weighted_mean(flux, dflux))

    def normalized_flux_std(self):
        """Measure standard deviation of flux in griz"""

        normalized_flux_std = lambda flux, mean: np.std(flux / mean, ddof=1)

        flux = getattr(self, "df").flux
        mean = getattr(self, 'mean')
        setattr(self, 'std', normalized_flux_std(flux, mean))

    def normalized_amplitude(self):
        """Measure the normalized amplitude of variations in griz"""

        normalized_amplitude = lambda flux, mean: (np.max(flux) - np.min(flux)) / mean

        flux = getattr(self, "df").flux
        mean = getattr(self, 'mean')
        setattr(self, 'amp', normalized_amplitude(flux, mean))

    def normalized_MAD(self):
        """Measure normalized Median Absolute Deviation (MAD) in griz"""

        normalized_mad = lambda flux, mean: np.median(np.abs((flux - np.median(flux))/mean))

        flux = getattr(self, "df").flux
        mean = getattr(self, 'mean')
        setattr(self, 'mad', normalized_mad(flux, mean))

    def beyond_1std(self):
        """Measure fraction of flux measurements beyond 1 std"""

        beyond_1std = lambda flux, mean: sum(np.abs(flux - mean) > np.std(flux, ddof = 1))/len(flux)

        flux = getattr(self, "df").flux
        mean = getattr(self, 'mean')
        setattr(self, 'beyond', beyond_1std(flux, mean))

    def skew(self):
        """Measure the skew of the flux measurements"""

        skew_l = lambda flux: stats.skew(flux)

        flux = getattr(self, "df").flux
        setattr(self, 'skew_v', skew_l(flux))

    def to_csv(self):
        df = pd.read_csv('dataset.csv')
        filename, file_extension = os.path.splitext(self.path)
        values = [
            filename, self.power, self.period, self.mean, self.std,
            self.amp, self.mad, self.beyond, self.skew_v]
        df.loc[len(df)] = values 
        df.to_csv('dataset.csv')
