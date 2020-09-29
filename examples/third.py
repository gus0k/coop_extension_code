
import sys
import time
import os
import json
import dill
import pickle
import numpy as np
from src.newdist import main_dist
from src.game import *
from pathlib import Path
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

if __name__ == '__main__':

    import sys
    parameters = sys.argv[1:]
    if len(parameters) < 5:
        quit()
# N = 2
# W = 1
# T = 48
# D = 1
# cant_bats = 0
# cant_solar = 0
# real_data = 30
# seed = 1234
# cost_solar = 109

    #parameters = [N, W, T, D, cant_bats, real_data, seed, cost_solar]
    string = '-'.join(map(str, parameters))  

    file_ =  Path(SIMULATION_PARAMETERS) / string

    N, W, T, D, cant_bats, cant_solar, real_data, seed, cost_solar, bpt, bp1, bp2, bp3, sp, size_bat, init_bat, ram_bat, ec_bat, ed_bat, cost_bat, max_solar, name = parameters

    N = int(N)
    W = int(W)
    T = int(T)
    D = int(D)
    cant_bats = int(cant_bats)
    cant_solar = int(cant_bats)
    real_data = int(real_data)
    seed = int(seed)
    cost_solar = int(cost_solar)
    bpt = int(bpt)
    bp1 = float(bp1)
    bp2 = float(bp2)
    bp3 = float(bp3)
    sp = float(sp)
    size_bat = float(size_bat)
    init_bat = float(init_bat)
    ram_bat = float(ram_bat)
    ec_bat = float(ec_bat)
    ed_bat = float(ed_bat)
    cost_bat = float(cost_bat)
    max_solar = float(max_solar)
    name = str(name)

    # N, W, T, D, cant_bats, cant_solar, real_data, seed, cost_solar, bpt, bp1, bp2, bp3, sp, size_bat, init_bat, cost_bat = map(int, [N, W, T, D, cant_bats, cant_solar, real_data, seed, cost_solar, bpt, bp1, bp2, bp3, sp, size_bat, init_bat, cost_bat])

    # ram_bat, ec_bat, ed_bat, max_solar = map(float, [ram_bat, ec_bat, ed_bat, max_solar])


    if file_.exists():
        print('Exists')
        quit()
    else:
        print('No exists')




    if bpt == 2:
        buying_price = np.ones(T) * bp1
        buying_price[: 14] = bp2
        buying_price[-2:] = bp2

    selling_price = np.ones(T) * sp


    ######

    # Obtain empty parameters
    res = paramteres_skeleton(N, T, W, D)
    player_info, battery_info, scenarios_training_load, scenarios_training_solar, probabilities, real_load, real_forecast, real_solar, forecast_solar = res

    battery_info = {
         'size': size_bat,
         'init': init_bat,
         'ram': ram_bat,
         'ec': ec_bat,
         'ed': ed_bat,
         'cost': cost_bat} # Cost 5000, 100 years payback


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
    players_with_solar = r.choice(range(N), size=cant_solar, replace=False)

    players = {}
    for n, k in enumerate(player_ids):
        has_solar = n in players_with_solar
        DFS = dfs_solar if has_solar else dfs_nosolar
        load_ = get_data(k, real_data, W + D, DFS[0])
        forecast_ = get_data(k, real_data, W + D, DFS[1])

        for i in range(0, 48 * W, 48):
            scenarios_training_load[n][i // 48, :] = load_[i: i + 48]
        for i in range(48 * W, 48 * (W + D), 48):
            real_load[n][(i // 48) - W, :] = load_[i: i + 48]
            real_forecast[n][(i // 48) - W, :] = forecast_[i: i + 48]

        if n in players_with_bats:
            player_info[n]['sm'] = size_bat # Tesla
            player_info[n]['ram'] = ram_bat
            player_info[n]['ec'] = ec_bat
            player_info[n]['ed'] = ed_bat

    ### Generate the solar data

    ## Training
    gen = r.uniform(max_solar, 0, size=(W, 24))
    solar_train = np.hstack([
        np.zeros((W, 12)), gen, np.zeros((W, 12))])
    scenarios_training_solar[:, :] = solar_train


    ## Validation

    gen = r.uniform(max_solar, 0, size=(D, 24))
    solar_validation = np.hstack([
        np.zeros((D, 12)), gen, np.zeros((D, 12))])
    real_solar[:, :] = solar_validation

    tmpsolar = np.vstack([scenarios_training_solar, real_solar])
    for d in range(D): 
        forecast_solar[d, :] = tmpsolar[:W + d, :].mean(axis=0)
        

    start = time.perf_counter()   

    ### Simple checks
    results = solve_one_game(
            N, T, D, W,
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

    elapsed = time.perf_counter() - start

    res_str = dict((str(k), v) for k, v in results.items())

    output = [
            N, T, D, W,
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
            elapsed,
            results,
    ]

    print('Elapsed time: ', round(elapsed, 2))

    with open(file_, 'wb') as fh:
        dill.dump(output, fh)


