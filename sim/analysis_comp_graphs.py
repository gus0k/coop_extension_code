import pickle
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

import os
import sys
from pathlib import Path

from sim.constants import OUTDIR_small

name = sys.argv[1]
params = pd.read_csv('sim/params_{}.csv'.format(name))


params['game'] = None
params['time_dist'] = 0
params['iters'] = 0
for i, r in params.iterrows():
    fn = '{}_{}_{}_{}_{}.pkl'.format(name, r.N, r['T'], r.G.strip(), r.S)
    try:
        with open(OUTDIR_small / fn, 'rb') as fh: data = pickle.load(fh)
        params.loc[i, 'game'] = data[0]
        params.loc[i, 'time_dist'] = data[4]
        params.loc[i, 'iters'] = data[-1]
    except:
        pass

params = params.dropna()

params['G'] = params['G'].map(str.strip)
params = params.sort_values(['N', 'time_dist'])
params['time_cent'] = params.game.map(lambda x: x.time_solve_fast)



from string import Template

figure1 = Template("""
\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
x tick label style={
/pgf/number format/1000 sep=},
xtick={ $xtick },
xticklabels={ $xtlabel },
yticklabel style={
        /pgf/number format/fixed,
        /pgf/number format/precision=5
},
scaled y ticks=false,
ylabel= $ylabel,
xlabel= $xlabel,
legend style={at={(0.5,-0.25)},
anchor=north,legend columns=-1},
ybar,
bar width=2pt,
ylabel near ticks,
width=7.5cm,
height=6cm,
]
\\addplot coordinates { $chord };
\\addplot coordinates { $regularval };
\\addplot coordinates { $wheelval };
\\addplot coordinates { $complete };
\\addplot coordinates { $path };
\\addplot coordinates { $tree };
\\addplot[black, fill=black] coordinates { $cycle };

\\legend{ $$E_n$$, $$R_{4, n}$$, $$W_n$$, $$K_n$$, $$P_n$$, $$T_n$$, $$C_n$$}
\\end{axis}
\\end{tikzpicture}
\\end{document}
""")

## Filling the plot

tmp = params.groupby(['G', 'N']).time_dist.mean()

xtlabel_ = '{' + ', '.join(map(str, params.N.unique())) + '}'
xtick_ = '{' + ', '.join(map(str, range(len(params.N.unique())))) + '}'

t = figure1.safe_substitute(
    xtick=xtick_,
    xtlabel=xtlabel_,
    ylabel='Elapsed time (seconds)',
    xlabel='Number of players',
    chord=' '.join(map(str, [x for x in enumerate(tmp['chordal'].values)])),
    regularval=' '.join(map(str, [x for x in enumerate(tmp['regular'].values)])),
    wheelval=' '.join(map(str, [x for x in enumerate(tmp['wheel'].values)])),
    complete=' '.join(map(str, [x for x in enumerate(tmp['complete'].values)])),
    path=' '.join(map(str, [x for x in enumerate(tmp['path'].values)])),
    tree=' '.join(map(str, [x for x in enumerate(tmp['tree'].values)])),
    cycle=' '.join(map(str, [x for x in enumerate(tmp['cycle'].values)])),
)

with open(OUTDIR_small / '{0}_comptop.tex'.format(name), 'w') as fh: fh.write(t)

#### Figure 2, iterations

figure2 = Template("""
\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
x tick label style={
/pgf/number format/1000 sep=},
xtick={ $xtick },
xticklabels={ $xtlabel },
yticklabel style={
        /pgf/number format/fixed,
        /pgf/number format/precision=5
},
scaled y ticks=false,
ylabel= $ylabel,
xlabel= $xlabel,
legend style={at={(0.5,-0.25)},
anchor=north,legend columns=-1},
ybar,
bar width=2pt,
ylabel near ticks,
width=7.5cm,
height=6cm,
]
\\addplot coordinates { $chord };
\\addplot coordinates { $regularval };
\\addplot coordinates { $wheelval };
\\addplot coordinates { $complete };
\\addplot coordinates { $path };
\\addplot coordinates { $tree };
\\addplot[black, fill=black] coordinates { $cycle };

\\legend{ $$E_n$$, $$R_{4, n}$$, $$W_n$$, $$K_n$$, $$P_n$$, $$T_n$$, $$C_n$$}
\\end{axis}
\\end{tikzpicture}
\\end{document}
""")

## Filling the plot

tmp2 = params.groupby(['G', 'N']).iters.mean()

t2 = figure2.safe_substitute(
    xtick=xtick_,
    xtlabel=xtlabel_,
    ylabel='Number of iterations before convergence',
    xlabel='Number of players',
    chord=' '.join(map(str, [x for x in enumerate(tmp2['chordal'].values)])),
    regularval=' '.join(map(str, [x for x in enumerate(tmp2['regular'].values)])),
    wheelval=' '.join(map(str, [x for x in enumerate(tmp2['wheel'].values)])),
    complete=' '.join(map(str, [x for x in enumerate(tmp2['complete'].values)])),
    path=' '.join(map(str, [x for x in enumerate(tmp2['path'].values)])),
    tree=' '.join(map(str, [x for x in enumerate(tmp2['tree'].values)])),
    cycle=' '.join(map(str, [x for x in enumerate(tmp2['cycle'].values)])),
)

with open(OUTDIR_small / '{0}_niters.tex'.format(name), 'w') as fh: fh.write(t2)
