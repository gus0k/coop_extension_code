
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

from examples.basic_example import solve_iterated_game, extract_core

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

for stl in scenarios_training_load:
    assert stl.shape == (W, T)

assert scenarios_training_solar.shape == (W, T)

for rl in real_load:
    assert rl.shape == (D, T)

for rf in real_forecast:
    assert rl.shape == (D, T)

assert real_solar.shape == (D, T)
assert forecast_solar.shape == (D, T)

results = dict()
for S in [[0], [1], [0, 1]]:

    players = []
    for n in S:
        pl_ = Player(x = scenarios_training_load[n],
                    sm = player_info[n].get('sm'),
                    s0 = player_info[n].get('s0'),
                    ram = player_info[n].get('ram'),
                    ec = player_info[n].get('ec'),
                    ed = player_info[n].get('ed'),
                    )
        players.append(pl_)

    pv_info = {
        'cost': cost_solar,
        'data': scenarios_training_solar.copy(),
    }

    ### Analysis no investment

    no_investment = solve_centralized(
            players,
            buying_price,
            selling_price,
            battery_info,
            pv_info,
            probabilities,
            0,
            0,
    )

    cost_no_investment = - no_investment[0].objective.value()

    ### Analysis investment

    investment = solve_centralized(
            players,
            buying_price,
            selling_price,
            battery_info,
            pv_info,
            probabilities,
            None,
            None,
    )

    cost_investment = - investment[0].objective.value()
    optimal_battery = investment[1]['batsizeXX'].varValue
    optimal_pv = investment[1]['pvsizeXX'].varValue

    core_investment = extract_core(len(S), investment)

    ### Analysis iterated

    cost_iterated_investment = np.zeros(D)
    cost_iterated_no_investment = np.zeros(D)
    cost_perfect_data_investment = np.zeros(D)

    cores_perfect_data = np.zeros((D, len(S)))

    for d in range(D):

        loads = [real_load[n][d, :] for n in S] + [real_solar[d, :]]
        loads = np.vstack(loads)

        forecasts = [real_forecast[n][d, :] for n in S] + [forecast_solar[d, :]]
        forecasts = np.vstack(forecasts)
        
        iter_investment = solve_iterated_game(
                len(S),
                players,
                loads,
                forecasts,
                buying_price,
                selling_price,
                battery_info,
                pv_info,
                optimal_battery,
                optimal_pv,
                )

        cost_iterated =  iter_investment[0].sum() + optimal_pv * cost_solar
        cost_iterated += optimal_battery * battery_info['cost']
        cost_iterated_investment[d] = cost_iterated

        iter_no_investment = solve_iterated_game(
                len(S),
                players,
                loads,
                forecasts,
                buying_price,
                selling_price,
                battery_info,
                pv_info,
                0,
                0,
                )

        cost_iterated_no =  iter_no_investment[0].sum() 
        cost_iterated_no_investment[d] = cost_iterated_no

        ### Finding core of iterated
            
        players = []
        for n in S:
            pl_ = Player(x = real_load[n][d, :].reshape(1, -1),
                        sm = player_info[n].get('sm'),
                        s0 = player_info[n].get('s0'),
                        ram = player_info[n].get('ram'),
                        ec = player_info[n].get('ec'),
                        ed = player_info[n].get('ed'),
                        )
            players.append(pl_)

        pv_info = {
            'cost': cost_solar,
            'data': real_solar[d, :].copy().reshape(1, -1),
        }

        ### Analysis no investment

        perfect_data = solve_centralized(
                players,
                buying_price,
                selling_price,
                battery_info,
                pv_info,
                np.ones(1),
                optimal_battery,
                optimal_pv,
        )

        cost_perfect_data = - perfect_data[0].objective.value()
        cost_perfect_data_investment[d] = cost_perfect_data

        core_perfect_data = extract_core(len(S), perfect_data)
        cores_perfect_data[d, :] = core_perfect_data



    result = {
    'cost_no_investment': cost_no_investment,
    'cost_investment': cost_investment,
    'optimal_battery': optimal_battery,
    'optimal_pv': optimal_pv,
    'cost_iterated_investment': cost_iterated_investment,
    'cost_iterated_no_investment': cost_iterated_no_investment,
    'core_investment': core_investment,
    'cores_perfect_data': cores_perfect_data,
    'cost_perfect_data_investment': cost_perfect_data_investment,
    }

    results[tuple(S)] = result
