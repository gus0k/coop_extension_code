import pandas as pd
import numpy as np

import dill
from pathlib import Path

NAME = '*test4'

pth = Path('~/Simulations/coop_extension')

flist = pth.expanduser().glob(NAME)


data = []
for fl in flist:
    with open(fl, 'rb') as fh:
        d = dill.load(fh)
        data.append(d)
D = len(data)

dataset = np.zeros((D, 7))
NN = tuple(range(40))

for i, dt in enumerate(data):
    dataset[i][0] = dt[-1]['roi_coop_theo'] # Theo roi coop
    dataset[i][1] = dt[-1]['roi_coop_days'].mean() # Mean roi coop
    dataset[i][2] = 100 * ((dataset[i, 0] / dataset[i , 1]) - 1)

    dataset[i][3] = dt[-1]['roi_hardware_theo'] # Theo roi hard
    dataset[i][4] = dt[-1]['roi_hardware_days'].mean() # Mean roi hard
    dataset[i][5] = 100 * ((dataset[i, 3] / dataset[i , 4]) - 1)

    dataset[i][6] = (dt[-1]['roi_coop_days'] / dt[-1][NN]['cost_iterated_investment']).mean() * 100


cnames = [
        'ROI Coop T',
        'ROI Coop E',
        '%ROI Coop Change',
        'ROI Hrdw T',
        'ROI Hrdw E',
        '%ROI Hrdw Change',
        '%ROI coop of total cost E',
]


df = pd.DataFrame(dataset, columns=cnames)
with open('table_{}.html'.format(NAME[1:]), 'w') as fh:
    fh.write(df.round(3).to_html())

