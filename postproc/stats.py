import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import dill
from pathlib import Path

NAME = '*test10'

pth = Path('~/Simulations/coop_extension')

flist = pth.expanduser().glob(NAME)


data = []
for fl in flist:
    with open(fl, 'rb') as fh:
        d = dill.load(fh)
        data.append(d)
D = len(data)

N = len([
    x for x in data[0][-1].keys() if isinstance(x, tuple)
    ]) - 1

dataset = np.zeros((D, 7))
NN = tuple(range(N))

for i, dt in enumerate(data):
    dataset[i][0] = dt[-1]['roi_coop_theo'] # Theo roi coop
    dataset[i][1] = dt[-1]['roi_coop_days'].mean() # Mean roi coop
    dataset[i][2] = 100 * ((dataset[i, 0] / dataset[i , 1]) - 1)

    dataset[i][3] = dt[-1]['roi_hardware_theo'] # Theo roi hard
    dataset[i][4] = dt[-1]['roi_hardware_days'].mean() # Mean roi hard
    dataset[i][5] = 100 * ((dataset[i, 3] / dataset[i , 4]) - 1)

    dataset[i][6] = (dt[-1]['roi_coop_days'] / dt[-1][NN]['cost_iterated_investment']).mean() * 100

#changes = np.zeros((D, N))
changes = []
j = 0
mean_red = np.zeros(D)
for i, dt in enumerate(data):
    coop = dt[-1]['final_core_payoffs'].sum(axis=0)
    single = np.array([dt[-1][(j,)]['cost_iterated_investment'].sum() for j in range(N)])
    mean_equal = (single.sum() - coop.sum()) * np.ones(N)
    # if not mean_equal:
    #     print(f'Reduction for {i}, not positive')

    mean_equal = mean_equal / np.abs(single)
    #mean_equal = ((single - (mean_equal / N)) / single)
    # mean_red[i] = mean_equal
    # print(i, mean_equal)
    change = 100 * ((single - coop) / np.abs(coop))
    #changes[i, :] = change
    for n in range(N):
        changes.append([int(i), n, mean_equal[n], 'Equal split'])
        changes.append([int(i), n, change[n], 'Core'])


chg = pd.DataFrame(changes)
chg.columns = ['day', 'player', 'change', 'Sharing mechanism']

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

import matplotlib as mpl
# mpl.use('pgf')
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))
sns.boxplot(data=chg[chg['Sharing mechanism'] == 'Equal split' ], x='day', y='change', palette="Blues", hue='Sharing mechanism', ax=ax[0])
ax[0].set_xticks([])
ax[0].set_xlabel('')
ax[0].set_ylabel('')
ax[0].get_legend().remove()
sns.boxplot(data=chg, palette="Blues", x='day', y='change', hue='Sharing mechanism', ax=ax[1])
ax[1].set_ylim([chg.change.min(), 20])
ax[1].axhline(y=0, linestyle='--')
ax[1].set_xlabel('Different simulations changing the load profiles')
ax[1].set_xticks([])
ax[1].set_ylabel('')

fig.text(0.04, 0.5, '% of change in the individual case versus the cooperative one.', va='center', rotation='vertical')
ax[1].legend(loc='lower left')
fig.tight_layout()
fig.savefig('/home/diego/github/thesis_phd/images/corevssplit.png')
fig.show()


# with open('table_{}.html'.format(NAME[1:]), 'w') as fh:
#     fh.write(df.round(3).to_html())

# with open('table_changes_{}.html'.format(NAME[1:]), 'w') as fh:
#     fh.write(chg.to_html())
