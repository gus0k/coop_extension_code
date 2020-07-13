import sys
import time
import os
import pickle
from src.newdist import main_dist
from src.game import *
from pathlib import Path
from constants import OUTDIR_large
from sim.params_alg import params

name = sys.argv[1]
params = params[name]

OUTDIR_large.mkdir(parents=True, exist_ok=True)

for N, T, G, seed, VF in params:
    path_file = OUTDIR_large / '{}_{}_{}_{}_{}.pkl'.format(name, N, T, G,seed)
    if os.path.isfile(path_file):
        print('File aready exits')
    else:
        start = time.time()
        g = generate_random_uniform(N, T, G, seed)
        g.init()
        x, gr, tim, niter = main_dist(g)
        end = time.time()
        costs = np.sum(x.mean(axis=0) * gr, axis=1)
        pc = g.get_payoff_core()
        if VF is True:
            _ = g.get_core_naive()
        data = [g, x, gr, end - start, tim.sum(), costs, pc, niter]
        with open(path_file, 'wb') as fh:
            pickle.dump(data, fh)
