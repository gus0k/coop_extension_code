from copy import deepcopy
from pathlib import Path
from sim.params_alg import params
from src.game import *
from src.newdist import main_dist
from src.player import Player

import numpy as np
import os
import pickle
import sys
import time

def extract_core(N, result):
    
    mo = result[0]
    real_cons = mo.constraints
    contributions = result[-1]

    payoff = np.zeros(N)
    for k in real_cons:
        du = real_cons[k].pi
        payoff -= contributions[k] * du

    return payoff


def solve_iterated_game(N, players_info, loads, forcast, pb, ps, batinfo, pvinfo, bat, pv, integer=False):

    bats = np.zeros(N + 1)
    T = loads.shape[1]
    costs = np.zeros(T)
    games = []

    for t in range(T):

        players = []
        for n in range(N):
            li = forcast[n, t:].copy()
            li[0] = loads[n, t]
            li = li.reshape(1, -1)
            pl = Player(x   = li, 
                        sm  = players_info[n]._sm,
                        s0  = bats[n],
                        ram = players_info[n]._ram,
                        ec  = players_info[n]._ec,
                        ed  = players_info[n]._ed)
            players.append(pl)

        batinfo['init'] = bats[N]

        pli = forcast[N, t:].copy()
        pli[0] = loads[N, t]
        pli = pli.reshape(1, -1)
        
        PV = {'cost': pvinfo['cost'], 'data': pli} 

        res = solve_centralized(players, pb[t:], ps[t:], batinfo, PV, np.array([1]), bat, pv, integer=integer)
        games.append(res)

        if res[0].status != 1:
            print('Solution not optimal')

        var_ = res[1]
        for n in range(N):
            ch = var_["chXX0_{0}_0".format(n)].varValue
            dis = var_["disXX0_{0}_0".format(n)].varValue
            bats[n] += ch - dis

        ch = var_["schXX0_0".format(n)].varValue
        dis = var_["sdisXX0_0".format(n)].varValue
        bats[N] += ch - dis

        zp = var_["zpXX0_0"].varValue
        zn = var_["znXX0_0"].varValue
        costs[t] = zp * pb[t] - zn * ps[t]

    return costs, games


def solve_one_game(N, T, D, W,
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
        integer,
        ):

    NN = tuple(range(N))

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
    for S in [[i] for i in range(N)] + [NN]:
    #for S in [[0, 1]]:

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
                integer=integer,
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
                integer=integer,
        )

        cost_investment = - investment[0].objective.value()
        optimal_battery = investment[1]['batsizeXX'].varValue
        optimal_pv = investment[1]['pvsizeXX'].varValue

        core_investment = extract_core(len(S), investment)
        assert np.allclose(core_investment.sum(), cost_investment)

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

        ### Payment for batteries
        bi = deepcopy(battery_info)
        bi['cost'] = 0
        pi = deepcopy(pv_info)
        pi['cost'] = 0
        investment_fixed = solve_centralized(
                players,
                buying_price,
                selling_price,
                bi,
                pi,
                probabilities,
                batfix = optimal_battery,
                pvfix = optimal_pv,
                proportions_core = core_investment,
                integer=False,
        )
        cost_fixed =  - investment_fixed[0].objective.value()
        core_investment_fixed = extract_core(len(S), investment_fixed)
        battery_core = core_investment - core_investment_fixed
        hardware_cost = optimal_pv * pv_info['cost'] + battery_info['cost'] * optimal_battery
        assert np.allclose(battery_core.sum(), hardware_cost)
        assert np.allclose(cost_fixed, core_investment_fixed.sum())

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
                    integer=integer,
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
                    integer=integer,
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

            bi = deepcopy(battery_info)
            bi['cost'] = 0
            bi['init'] = 0
            pi = deepcopy(pv_info)
            pi['cost'] = 0
            perfect_data = solve_centralized(
                    players,
                    buying_price,
                    selling_price,
                    bi,
                    pi,
                    np.ones(1),
                    optimal_battery,
                    optimal_pv,
                    proportions_core = None,
                    integer=False,
            )

            cost_perfect_data = - perfect_data[0].objective.value() + hardware_cost
            cost_perfect_data_investment[d] = cost_perfect_data

            core_perfect_data = extract_core(len(S), perfect_data) + battery_core
            cores_perfect_data[d, :] = core_perfect_data
            assert np.allclose(core_perfect_data.sum(), cost_perfect_data)



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


    ### Find theoretical ROI for cooperation
    roi_coop_theo = -results[NN]['cost_investment']
    for n in range(N): roi_coop_theo += results[(n, )]['cost_investment']

    ### Find theoretical ROI for hardware investment
    roi_hardware_theo = 0
    for n in range(N): roi_hardware_theo += results[(n, )]['cost_no_investment']
    for n in range(N): roi_hardware_theo -= results[(n, )]['cost_investment']

    final_core_payoffs = np.zeros((D, N))

    roi_coop_days = np.zeros(D)
    roi_hardware_days = np.zeros(D)


    for d in range(D):

        roi_coop_days[d] = -results[NN]['cost_iterated_investment'][d]
        for n in range(N):
            roi_coop_days[d] += results[(n, )]['cost_iterated_investment'][d]

        for n in range(N): roi_hardware_days[d] += results[(n, )]['cost_iterated_no_investment'][d]
        for n in range(N): roi_hardware_days[d] -= results[(n, )]['cost_iterated_investment'][d]

        gap_tmp = np.zeros(N)
        for k, v in results.items():
            gap = v.get('cost_iterated_investment')[d]
            gap -= v.get('cost_perfect_data_investment')[d]
            if len(k) == 1:
                gap_tmp[k[0]] = gap

        if not np.allclose(gap_tmp.sum(), 0):
            gap_tmp = gap_tmp / gap_tmp.sum() * gap
        else:
            gap_tmp = np.ones(N) / N * gap

        final_core_payoffs[d, :] = results[NN]['cores_perfect_data'][d]
        final_core_payoffs[d, :] += gap_tmp

    increased_storage = results[NN]['optimal_battery'] * battery_info['size']
    for n in range(N):
        increased_storage -= results[(n,)]['optimal_battery'] * battery_info['size']

    increased_pv = results[NN]['optimal_pv']
    for n in range(N):
        increased_pv -= results[(n,)]['optimal_pv']


    results['final_core_payoffs'] = final_core_payoffs
    results['roi_coop_theo'] = roi_coop_theo
    results['roi_hardware_theo'] = roi_hardware_theo
    results['roi_coop_days'] = roi_coop_days
    results['roi_hardware_days'] = roi_hardware_days
    results['increased_storage'] = increased_storage
    results['increased_pv'] = increased_pv

    return results

def paramteres_skeleton(N, T, W, D):

    one_player = {'sm': 0, 's0': 0, 'ram': 0, 'ec': 1, 'ed': 1}

    player_info = [deepcopy(one_player) for n in range(N)]

    battery_info = {
        'size': 0,
        'init': 0,
        'ram': 0,
        'ec': 1,
        'ed': 1,
        'cost': 0}


    scenarios_training_load = [np.zeros((W, T)) for n in range(N)]
    scenarios_training_solar = np.zeros((W, T))

    probabilities = np.ones(W) / W


    real_load = [np.zeros((D, T)) for n in range(N)]
    real_forecast = [np.zeros((D, T)) for n in range(N)]

    real_solar = np.zeros((D, T))
    forecast_solar = np.zeros((D, T))

    res = (player_info, battery_info, scenarios_training_load,
        scenarios_training_solar, probabilities, real_load, real_forecast,
        real_solar, forecast_solar)

    return res
