#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results_fast.csv', names=['ADC_READOUTS'])
adc_readouts = df['ADC_READOUTS']

fig, ax = plt.subplots(1, 1)

for width in [5, 10, 20, 50, 100]:
    window = adc_readouts.rolling(width).mean()
    ax.hist(window, bins=100, log=True,
            histtype='step', label=f'width {width}, stddev {window.std():.2f}')


ax.legend()
ax.set_xlabel('ADC readout', loc='right')
fig.savefig('rolling_window.png', dpi=300)
