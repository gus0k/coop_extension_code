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







    
