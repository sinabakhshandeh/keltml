import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import scipy.stats as st
import os, sys
from astropy.timeseries import BoxLeastSquares
from astropy import units as u

class Kelt:
	
	KELT_BINS = 250

	def __init__(self, path, t0 = 0):
		self.path = path
		self.t0 = t0
		self.data = np.loadtxt(path)
		self.df = pd.DataFrame({'t': self.data[:, 0], 'm': self.data[:, 1], 'e': self.data[:, 2]})
		# self.df['m_median'] = self.df.m - np.median(self.df.m)
		self.julian_day_fixer()
		self.fig, [self.ax1, self.ax2, self.ax3] = plt.subplots(3, 1, figsize=(6, 6))
		self.ax1.set_title('Full Light Curve')
		self.plot_()

	def plot_(self):
		self.ax1.plot((self.df.t), (self.df.m), 'b.')
		self.ax1.set_ylim(max(self.df.m), min(self.df.m))
		self.ax1.set_ylabel('Magnitude')
		self.ax1.set_xlabel('Time(days)')

		

	def plot_ls(self):
		
		
		self.ax3.plot(1./self.frequency, self.power)
		self.ax3.set_ylabel("Power")
		self.ax3.set_xlabel("Period(days)")
		self.ax3.set_xlim(self.pmin,self.pmax)
		self.ax3.set_ylim(0.9*min(self.power), 1.05*max(self.power))
		self.ax3.axvline(self.period, color = 'k')
		self.ax3.set_title(self.method +' Periodogram')
		# fig.set_size_inches(12.0,10.0)
		


	def julian_day_fixer(self):
		if self.df.t[0] > 10000:
			self.df['t_kelt'] = self.df.t - 2450000
		else:
			self.df['t_kelt'] = self.df.t

	def t_0_fixer(self):
		self.phase0 = (self.t0 / self.period) - np.fix(self.t0 / self.period)
		self.phase = self.phase - self.phase0 - 0.25
		for i in np.arange(len(self.phase)):
			if self.phase[i] < 0:
				self.phase[i] += 1
			if self.phase[i] > 1:
				self.phase[i] -= 1

	def period_finder(self, nt = 5, min_p=1, max_p=100, n_f = 10000, auto = True, method = 'LS_astropy'):

		self.pmin = min_p
		self.pmax = max_p
		self.method = method

		if self.method == 'LS_astropy':
			if auto:
				ls = LombScargle(self.df.t.values, self.df.m.values, self.df.e.values, nterms=nt)
				self.frequency, self.power = ls.autopower(minimum_frequency=1./self.pmax, maximum_frequency=1./self.pmin)
			else:
				self.frequency = np.linspace(1./self.pmax, 1./self.pmin, n_f)
				self.power = LombScargle(self.df.t.values, self.df.m.values, self.df.e.values).power(self.frequency)

		elif self.method == 'BLS_astropy':
			model = BoxLeastSquares(self.df.t.values * u.day, self.df.m.values, dy=self.df.e.values)
			if auto:
				periodogram = model.autopower(0.2)
				self.frequency = 1./periodogram.period
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

		self.set_period()
		self.plot_ls()


	def set_period(self):
		period = (1. / self.frequency[np.argmax(self.power)])

		if period-np.fix(period) <0.009:
			self.period = (1. / self.frequency[(np.asarray(self.power).argsort()[-2])])
		else:
			self.period = period

		self.set_phase()

	def set_phase(self):
		self.phase = (self.df.t_kelt / self.period) - np.fix(self.df.t_kelt / self.period)
		self.t_0_fixer()

	def plot_phased(self):
		self.ax2.plot(self.phase, self.df.m, 'y.', markersize=5)
		self.ax2.plot(self.bin_middles_kelt, self.med_stat_kelt, 'k', markersize=6)
		self.ax2.plot(self.phase+1, self.df.m, 'y.', markersize=5)
		self.ax2.plot(np.asarray(self.bin_middles_kelt)+1, self.med_stat_kelt, 'k', markersize=6)
		self.ax2.set_xlim(0,2)
		self.ax2.set_ylim(self.max_m, self.min_m)
		self.ax2.set_ylabel('Magnitude')
		self.ax2.set_xlabel('Phase')
		self.ax2.set_title('Phased Light Curve with Period = '+str(round(self.period, 3)) +' days')
		self.fig.tight_layout()
		self.fig.savefig(str(self.path)+'_'+self.method+'.png')

	def binned_statistic(self):
		self.max_m = ((30 * (max(self.df.m) - min(self.df.m))) / 100) + min(self.df.m)
		self.min_m = ((10 * (max(self.df.m) - min(self.df.m))) / 100) + min(self.df.m)

		(self.med_stat_kelt, bin_edges, binnumber) = st.binned_statistic(self.phase, self.df.m,
																		 statistic='median', bins=Kelt.KELT_BINS)
		r = (bin_edges[1] - bin_edges[0]) / 2

		self.bin_middles_kelt = [bin_edges[index] + r for index in np.arange(len(bin_edges) - 1)]
		self.plot_phased()
