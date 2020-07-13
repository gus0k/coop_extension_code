import sys
import time
import os
import pickle
from src.newdist import main_dist
from src.game import *
from pathlib import Path
from sim.constants  import OUTDIR_small

OUTDIR_small.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':

    if len(sys.argv) < 5:
        sys.exit()
    N = int(sys.argv[1])
    T = int(sys.argv[2])
    G = sys.argv[3].strip()
    seed = int(sys.argv[4])

    path_file = OUTDIR_small / '{}_{}_{}_{}.pkl'.format(N, T, G,seed)

    if os.path.isfile(path_file):
        print('File aready exits')
        sys.exit()
    else:
        start = time.time()
        g = generate_random_uniform(N, T, G, seed)
        g.init()
        x, gr, tim, niter = main_dist(g)
        end = time.time()
        costs = np.sum(x.mean(axis=0) * gr, axis=1)
        pc = g.get_payoff_core()
        data = [g, x, gr, end - start, tim.sum(), costs, pc, niter]

        with open(path_file, 'wb') as fh:
            pickle.dump(data, fh)
