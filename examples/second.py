
import sys
import time
import os
import pickle
import numpy as np
from src.newdist import main_dist
from src.game import *
from pathlib import Path
from sim.params_alg import params
from src.player import Player
from copy import deepcopy

from src.extensiongame import *

player_info = [
        {'sm': 0,
         's0': 0,
         'ram': 1,
         'ec': 1,
         'ed': 1,
        },
        {'sm': 0,
         's0': 0,
         'ram': 1,
         'ec': 1,
         'ed': 1,
        },
]

cost_solar = 2

battery_info = {
    'size': 1,
    'init': 0,
    'ram': 1,
    'ec': 1,
    'ed': 1,
    'cost': 1
}

N = 2 # Number of players
T = 2 # Number of time-slots
W = 2 # Number of scenarios used
D = 1 # Number of days in which the problem is evaluated
NN = tuple(range(N))

scenarios_training_load = [
    np.array([
        [0, 0.2],
        [0, 1.0],
    ]),
    np.array([
        [0, 1],
        [0, 0.5],
    ])
]


scenarios_training_solar = np.array([
    [-1, 0.0],
    [-1, 0.0]
])

buying_price = np.ones(T) * 4.0
selling_price = np.ones(T) * 0.1

probabilities = np.ones(W) / W

real_load = [
    np.array([
        [0, 0.2],
    ]),
    np.array([
        [0, 1.0],
    ]),
]

real_forecast = [
    np.array([
        [0, 0.2],
    ]),
    np.array([
        [0, 0.8],
    ]),
]

real_solar = np.array([[-1, 0]])
forecast_solar = np.array([[-1, 0]])

### Simple checks
results = solve_one_game(N, T, D, W,
        buying_price,
        selling_price,
        real_load,
        real_forecast,
        real_solar,
        forecast_solar,
        scenarios_training_load,
        scenarios_training_solar,
        player_info,
        battery_info,
        cost_solar,
        probabilities,
        integer=False,
        )
print(results)
