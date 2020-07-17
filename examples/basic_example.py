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

#T = 2
#N = 2
#player_list = []

#batinfo = {
#    'size': 1,
#    'init': 0,
#    'ram': 1,
#    'ec': 1,
#    'ed': 1,
#    'cost': 1
#}

#pvinfo = {
#    'cost': 2,
#    'data': np.array([[-1, 0], [-1, 0]]),
#}

#probabilities = np.array([0.5, 0.5])

#p1 = Player(x=np.array([[0, 0.2], [0, 1]]),
#            sm = 0,
#            s0 = 0,
#            ram = 1,
#            ec = 1,
#            ed = 1)
#player_list.append(p1)

#p2 = Player(x=np.array([[0, 1], [0, 0.5]]),
#            sm = 0,
#            s0 = 0,
#            ram = 1,
#            ec = 1,
#            ed = 1)
#player_list.append(p2)


#buying_price = np.array([4, 4]) 
#selling_price = np.ones(T) * 0.1

#G = nx.complete_graph(N)

#game = Game(player_list, buying_price, selling_price, G, batinfo, pvinfo, probabilities)

##game.init()
#game.solve()

#mo = game._model
#real_cons = mo.constraints
#contributions = game._res[-1]
#mo.writeLP('model.lp')
#with open('model.lp', 'r') as fh: mostr = fh.readlines()

#L = len(mostr)
#duals = np.zeros(L)

#pairs = []

#for l, row in enumerate(mostr):
#    if row[:4] == 'cons':
#        con_name = row.split(':')[0]
#        du = real_cons[con_name].pi
#        duals[l] = du
#        contrib = contributions[con_name]
#        pr = (con_name, du, contrib)
#        pairs.append(pr)


#payoff = np.zeros(N)
#for cn, du, ctr in pairs:
#   # print(cn, '---', du, '---', ctr, '---', du * ctr)
#    payoff += du * ctr

#print(payoff)

def extract_core(N, result):
    
    mo = result[0]
    real_cons = mo.constraints
    contributions = result[-1]

    payoff = np.zeros(N)
    for k in real_cons:
        du = real_cons[k].pi
        payoff -= contributions[k] * du

    return payoff


def solve_iterated_game(N, players_info, loads, forcast, pb, ps, batinfo, pvinfo, bat, pv):

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

        res = solve_centralized(players, pb[t:], ps[t:], batinfo, PV, np.array([1]), bat, pv)
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


# loads = np.array([
#     [0, .2],
#     [0, 1],
#     [-1, 0], # Load PV
#     ])


# forecast = np.array([
#     [0, .2],
#     [0, 0.8],
#     [-1, 0], # Load PV
#     ])

# res2 =  solve_iterated_game(2, player_list, loads, forecast, buying_price, selling_price, batinfo, pvinfo, 1.2, 1.2)



#game = Game(player_list, buying_price, selling_price, G, batinfo, pvinfo, probabilities)







    
