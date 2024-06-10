#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spopt


# Define statistical distributions

def gaussian(x, normalisation, mean, stddev):
    return normalisation/(stddev*math.sqrt(2*math.pi)) * np.exp(-0.5 * ((x - mean)/stddev)**2)


def double_gaussian(x, norm1, mean1, stddev1, norm2, mean2, stddev2):
    return gaussian(x, norm1, mean1, stddev1) + gaussian(x, norm2, mean2, stddev2)


def cauchy(x, normalisation, x_0, gamma):
    return normalisation/math.pi * gamma/((x - x_0)**2 + gamma**2)


def gaussian_plus_cauchy(x, norm_g, mean, stddev, norm_c, x_0, gamma):
    return gaussian(x, norm_g, mean, stddev) + cauchy(x, norm_c, x_0, gamma)


# load data and make scatterplot
df = pd.read_csv('results_fast.csv', names=['ADC_READING'])

fig, ax = plt.subplots(1, 1)
ax.plot(df['ADC_READING'][:1000], marker='.', linestyle='none')
fig.savefig('results_fast.png', dpi=300)


# create histogram of ADC readouts
fig, ax = plt.subplots(1, 1)
bin_values, bin_edges, patches = ax.hist(df['ADC_READING'], bins=100, log=True)
bin_centers = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
x_for_function = np.linspace(bin_edges[0], bin_edges[-1], 1000)


# fit gaussian to histogram
popt, pcov, info, mesg, ier = spopt.curve_fit(
    gaussian, xdata=bin_centers, ydata=bin_values,
    p0=(bin_values[50], df['ADC_READING'].mean(), df['ADC_READING'].std()),
    full_output=True
)
print(f'''
### Gaussian Fit Results ###
mean: \t {popt[1]}
std: \t {popt[2]}
norm: \t {popt[0]}
{mesg}
''')
ax.plot(x_for_function, gaussian(x_for_function, *popt),
        marker='', linestyle='-', label='gaussian')


# fit sum of two gaussians to histogram
popt, pcov, info, mesg, ier = spopt.curve_fit(
    double_gaussian, xdata=bin_centers[10:90], ydata=bin_values[10:90],
    p0=(bin_values[50], df['ADC_READING'].mean(), df['ADC_READING'].std(),
        20, df['ADC_READING'].mean(), 60),
    bounds=(0, np.inf),
    method='dogbox',
    full_output=True
)
print(f'''
### Double Gaussian Fit Results ###
mean: \t {popt[1]} \t {popt[4]}
std: \t {popt[2]} \t {popt[5]}
norm: \t {popt[0]} \t {popt[3]}
{mesg}
''')
ax.plot(x_for_function, double_gaussian(x_for_function, *popt),
        marker='', linestyle='-', label='double gaussian')


# fit a cauchy distribution to histogram
popt, pcov, info, mesg, ier = spopt.curve_fit(
    cauchy, xdata=bin_centers, ydata=bin_values,
    p0=(bin_values[50], df['ADC_READING'].mean(), 200),
    bounds=(-np.inf, np.inf),
    full_output=True
)
print(f'''
### Cauchy Fit Results ###
x_0: \t {popt[1]}
HWHM: \t {popt[2]}
norm: \t {popt[0]}
{mesg}
''')
ax.plot(x_for_function, cauchy(x_for_function, *popt),
        marker='', linestyle='-', label='cauchy')


# fit sum of gaussian and cauchy distribution to histogram
popt, pcov, info, mesg, ier = spopt.curve_fit(
    gaussian_plus_cauchy, xdata=bin_centers, ydata=bin_values,
    p0=(bin_values[50], df['ADC_READING'].mean(), df['ADC_READING'].std(),
        20, df['ADC_READING'].mean(), 200),
    bounds=(0, np.inf),
    method='dogbox',
    full_output=True
)
print(f'''
### Gaussian plus Cauchy Fit Results ###
mean: \t {popt[1]}
std: \t {popt[2]}
norm_g:\t {popt[0]}
x_0: \t {popt[4]}
HWHM: \t {popt[5]}
norm_c:\t {popt[3]}
{mesg}
''')
ax.plot(x_for_function, gaussian_plus_cauchy(x_for_function, *popt),
        marker='', linestyle='-', label='gaussian plus cauchy')


# Do some formatting and save histogram with fitted functions
ax.set_ylim(bottom=0.1, top=2*max(bin_values))
ax.legend()

fig.savefig('histogram.png', dpi=300)
