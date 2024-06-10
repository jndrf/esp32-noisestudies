import machine
import time

adc = machine.ADC(32, atten=3)

starttime = time.time()

readouts = []

while time.time() - starttime < 1:
    readouts.append(adc.read())

with open('results_fast.csv', 'w') as f:
    for r in readouts:
        f.write(str(r))
        f.write('\n')
