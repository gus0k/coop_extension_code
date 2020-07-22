
import sys
import time
import os
import json
import pickle
import numpy as np
from src.newdist import main_dist
from src.game import *
from pathlib import Path
from sim.params_alg import params
from src.player import Player
from copy import deepcopy
from src.extensiongame import *
from src.process_data import get_data

from config import *
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

###### Extra parameters

N = 20
W = 30
T = 48
D = 10
cant_bats = 5
real_data = 30
seed = 1234
cost_solar = 50

parameters = [N, W, T, D, cant_bats, real_data, seed, cost_solar]

string = '-'.join(map(str, parameters))

buying_price = np.ones(T) * 20.0
buying_price[: T // 2] = 15.0
selling_price = np.ones(T) * 1.0


######

# Obtain empty parameters
res = paramteres_skeleton(N, T, W, D)
player_info, battery_info, scenarios_training_load, scenarios_training_solar, probabilities, real_load, real_forecast, real_solar, forecast_solar = res

battery_info = {
     'size': 13,
     'init': 0,
     'ram': 2.5,
     'ec': 0.95,
     'ed': 0.95,
     'cost': 136} # Cost 5000, 100 years payback


# Load consumption profiles
r = np.random.RandomState(seed)
player_ids = r.choice(np.arange(126), N, replace=False)

data_original = pd.read_csv(DATA, index_col='date', parse_dates=True)
data_forcast = pd.read_csv(DATA_FORCAST, index_col='date', parse_dates=True)
dfs_nosolar = [data_original, data_forcast]

data_solar= pd.read_csv(DATA_SOLAR, index_col='date', parse_dates=True)
data_solar_forcast = pd.read_csv(DATA_SOLAR_FORCAST, index_col='date', parse_dates=True)
dfs_solar = [data_solar, data_solar_forcast]

# Initialize data

players_with_bats = r.choice(range(N), size=cant_bats, replace=False)

players = {}
for n in range(N):
    has_solar = n <= (N // 2)
    DFS = dfs_solar if has_solar else dfs_nosolar
    load_ = get_data(n, real_data, W + D, DFS[0])
    forecast_ = get_data(n, real_data, W + D, DFS[1])

    for i in range(0, 48 * W, 48):
        scenarios_training_load[n][i // 48, :] = load_[i: i + 48]
    for i in range(48 * W, 48 * (W + D), 48):
        real_load[n][(i // 48) - W, :] = load_[i: i + 48]
        real_forecast[n][(i // 48) - W, :] = forecast_[i: i + 48]

    if n in players_with_bats:
        player_info[n]['sm'] = 13 # Tesla
        player_info[n]['ram'] = 2.5
        player_info[n]['ec'] = 0.95
        player_info[n]['ed'] = 0.95

### Generate the solar data

## Training
gen = r.uniform(-0.3, 0, size=(W, 24))
solar_train = np.hstack([
    np.zeros((W, 12)), gen, np.zeros((W, 12))])
scenarios_training_solar[:, :] = solar_train

## Validation

gen = r.uniform(-0.3, 0, size=(D, 24))
solar_validation = np.hstack([
    np.zeros((D, 12)), gen, np.zeros((D, 12))])
real_solar[:, :] = solar_validation

tmpsolar = np.vstack([scenarios_training_solar, real_solar])
for d in range(D): 
    forecast_solar[d, :] = tmpsolar[:W + d, :].mean(axis=0)
    

        

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
        integer=True,
        )

res_str = dict((str(k), v) for k, v in results.items())

with open(string + '.json', 'w') as fh:
    fh.write(
        json.dumps(res_str, indent=4, cls=NumpyEncoder)
    )


