import pickle
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from string import Template

import matplotlib
matplotlib.use('TkAgg')

import os
import sys
from pathlib import Path

from sim.constants import OUTDIR_large

name = sys.argv[1]

if name == 'simple':
    seed = 3
elif name == 'complete':
    seed = 13


rows = []
files = OUTDIR_large.glob('*_{0}.pkl'.format(seed))
for fn in files:
    try:
        with open(fn, 'rb') as fh: data = pickle.load(fh)
        name_, N, T, G, S = fn.name[:-4].split('_')
        g = data[0]
        time_dist = data[4]
        n_iter = data[-1]

        tup = (N, G, S, g, time_dist, n_iter)
        rows.append(tup)
    except Exception as e:
        print(e)
    
df = pd.DataFrame(rows)
df.columns = ['N', 'G', 'S', 'game', 'dist', 'iters']
df['cent'] = df.game.map(lambda x: x.time_solve_fast)
df['naive'] = df.game.map(lambda x: x.time_core_naive)
df['N'] = df['N'].astype(int)
df = df.sort_values('N')

melt = pd.melt(df, id_vars=['N'], value_vars=['dist', 'cent', 'naive'])



figure = Template("""
\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\begin{document}

\\begin{tikzpicture}
\\begin{axis}[
ymode=log,
log ticks with fixed point,
height=4.5cm,
width=8cm,
xlabel=Number of players,
ylabel=Elapsed time (seconds),
ylabel near ticks,
cycle list name=color,
legend style={at={(0.5,-0.35)},
anchor=north,legend columns=-1},
]

\\addplot coordinates { $naive };
\\addplot coordinates { $cent };
\\addplot coordinates { $dist };

\legend{Naive, Centralized, Distributed};
\\end{axis}
\\end{tikzpicture}

\\end{document}
""")

naive_ = ' '.join(map(str, [(x, y) for x, y in df[['N','naive']].dropna().values]))
cent_ = ' '.join(map(str, [(x, y) for x, y in df[['N','cent']].dropna().values]))
dist_ = ' '.join(map(str, [(x, y) for x, y in df[['N','dist']].dropna().values]))

t = figure.safe_substitute(
    naive=naive_,
    cent=cent_,
    dist=dist_
)

with open(OUTDIR_large / '{}_compalg.tex'.format(name), 'w') as fh: fh.write(t)
