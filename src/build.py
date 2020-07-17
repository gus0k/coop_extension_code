import pulp as plp
import pandas as pd
import numpy as np
from copy import deepcopy
from src.player import Player

from collections import namedtuple


# def solve_centralized(player_list, bp, sp, batinfo, solarinfo):

#     mo = plp.LpProblem(name="model")
#     contributions = {}
#     set_N = range(len(player_list))
#     set_T = range(len(player_list[0]._x.shape[1]))
#     set_W = range(len(player_list[0]._x.shape[0]))

#     var = {
#         'zp': 'T',
#         'zn': 'T',
#         'ch': 'WNT',
#         'dis': 'WNT',
#         'sch': 'WT',
#         'sdh': 'WT',
#         'sbs': '1',
#         'sss': '1',
#     }

#     for k, v in var.items():

#         if v == 'T': # Time-slots
#             tmp = {t:
#                 plp.LpVariable(
#                     cat=plp.LpContinuous,
#                     lowBound=0,
#                     upBound=None,
#                     name="{0}_{1}".format(k, t))
#             for t in set_T}

#         elif v == 'WNT': # Scenario Players Time-slots

#             tmp = {(w, n, t):
#                 plp.LpVariable(
#                     cat=plp.LpContinuous,
#                     lowBound=0,
#                     upBound=None,
#                     name="{0}_{1}_{2}_{3}".format(k, w, n, t))
#             for t in set_T for n in set_N for w in set_W}

#         elif v == 'WT': # Scenarios, Time-slots
            
#             tmp = {(w, t):
#                 plp.LpVariable(
#                     cat=plp.LpContinuous,
#                     lowBound=0,
#                     upBound=None,
#                     name="{0}_{1}_{2}".format(k, w, t))
#             for t in set_T for w in set_W}

#         else:
#             tmp = plp.LpVariable(
#                     cat=plp.LpContinuous,
#                     lowBound=0,
#                     upBound=None,
#                     name=k)

#         var[k] = tmp

#     cons_bat_ub = {(n, j) :
#     plp.LpConstraint(
#                  e=plp.lpSum(ch_vars[(n, t)] - dis_vars[(n, t)] for t in range(j +
#                  1)),
#                  sense=plp.LpConstraintLE,
#                  rhs=player_list[n]._sm - player_list[n]._s0,
#                  name="cons_bat_up_{0}_{1}".format(n, j))
#            for j in set_T for n in set_N}

#     N = len(set_N)
#     for n in set_N:
#         for j in set_T:
#             cont = np.zeros(N)
#             cont[n] = player_list[n]._sm - player_list[n]._s0
#             contributions[cons_bat_ub[(n, j)].name] = cont




def solve_centralized(player_list, buying_price, selling_price, batinfo, pvinfo, prob, batfix=None, pvfix=None):

    model = plp.LpProblem(name="model")

    contributions = {}

    N = len(player_list)
    set_N = range(N)
    set_T = range(player_list[0]._x.shape[1])
    set_W = range(player_list[0]._x.shape[0])

    

    ## z^+_t
    zp_vars = {(w, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            upBound=None,
            name="zpXX{0}_{1}".format(w, t)
        ) for t in set_T for w in set_W
    }

    ## z^-_t
    zn_vars = {(w, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            upBound=None,
            name="znXX{0}_{1}".format(w, t)
        ) for t in set_T for w in set_W
    }

    ## c^n_t

    ch_vars = {(w, n, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            name="chXX{0}_{1}_{2}".format(w, n, t)
        ) for t in set_T for n in set_N for w in set_W
    }

    shared_ch_vars = {(w, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            name="schXX{0}_{1}".format(w, t)
        ) for t in set_T for w in set_W
    }
    ## d^n_t

    dis_vars = {(w, n, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            name="disXX{0}_{1}_{2}".format(w, n, t)
        ) for t in set_T for n in set_N for w in set_W
    }

    shared_dis_vars = {(w, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            name="sdisXX{0}_{1}".format(w, t)
        ) for t in set_T for w in set_W
    }

    batsize = plp.LpVariable(cat=plp.LpContinuous, lowBound=0, name="batsizeXX")
    pvsize = plp.LpVariable(cat=plp.LpContinuous, lowBound=0, name="pvsizeXX")


    var = [zp_vars, zn_vars, ch_vars, 
            dis_vars, shared_ch_vars, shared_dis_vars,
            batsize, pvsize]

    vars_ = {}
    for vi in var:
        if isinstance(vi, dict):
            for v in vi.values():
                vars_[v.name] = v
        else:
            vars_[vi.name] = vi

    #### Constraints

    ## Variable boudns

    ######## Start private batteries

    ## Private Battery upper bound
    cons_bat_ub = {(w, n, j) :
    plp.LpConstraint(
                 e=plp.lpSum(ch_vars[(w, n, t)] - dis_vars[(w, n, t)] for t in range(j +
                 1)),
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._sm - player_list[n]._s0,
                 name="cons_bat_up_{0}_{1}_{2}".format(w, n, j))
           for j in set_T for n in set_N for w in set_W}

    for n in set_N:
        for j in set_T:
            for w in set_W:
                cont = np.zeros(N)
                cont[n] = player_list[n]._sm - player_list[n]._s0
                contributions[cons_bat_ub[(w, n, j)].name] = cont

    for k in cons_bat_ub: model.addConstraint(cons_bat_ub[k])

    ## Private Battery lower bound
    cons_bat_lb = {(w, n, j) :
    plp.LpConstraint(
                 e=-plp.lpSum(ch_vars[(w, n, t)] - dis_vars[(w, n, t)] for t in range(j +
                 1)),
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._s0,
                 name="cons_bat_low_{0}_{1}_{2}".format(w, n, j))
           for j in set_T for n in set_N for w in set_W}

    for k in cons_bat_lb: model.addConstraint(cons_bat_lb[k])

    ### Contributions
    for n in set_N:
        for j in set_T:
            for w in set_W:
                cont = np.zeros(N)
                cont[n] = player_list[n]._s0
                contributions[cons_bat_lb[(w, n, j)].name] = cont


    ## Private Battery ramp up

    cons_bnd_ub = {(w, n, t) :
    plp.LpConstraint(
                 e=ch_vars[(w, n, t)],
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._ram,
                 name="cons_bnd_up_{0}_{1}_{2}".format(w, n, t))
           for t in set_T for n in set_N for w in set_W}

    for k in cons_bnd_ub: model.addConstraint(cons_bnd_ub[k])

    ### Contributions
    for n in set_N:
        for j in set_T:
            for w in set_W:
                cont = np.zeros(N)
                cont[n] = player_list[n]._ram
                contributions[cons_bnd_ub[(w, n, j)].name] = cont

    ## Private Battery lower bound
    cons_bnd_lb = {(w, n, t) :
    plp.LpConstraint(
                 e=dis_vars[(w, n, t)],
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._ram,
                 name="cons_bnd_low_{0}_{1}_{2}".format(w, n, t))
           for t in set_T for n in set_N for w in set_W}

    for k in cons_bnd_lb: model.addConstraint(cons_bnd_lb[k])

    ### Contributions
    for n in set_N:
        for j in set_T:
            for w in set_W:
                cont = np.zeros(N)
                cont[n] = player_list[n]._ram
                contributions[cons_bnd_lb[(w, n, j)].name] = cont

    ####### End private batteries

    ####### Start public battery

    ## Public battery upper bound
    cons_shared_bat_ub = {(w, j) :
    plp.LpConstraint(
                 e=plp.lpSum(shared_ch_vars[(w, t)] - shared_dis_vars[(w, t)] for t in range(j +
                 1)) - batinfo['size'] * batsize ,
                 sense=plp.LpConstraintLE,
                 rhs= - batinfo['init'],
                 name="cons_shared_bat_up_{0}_{1}".format(w, j))
           for j in set_T for w in set_W}

    for j in set_T:
        for w in set_W:
            cont = np.zeros(N)
            contributions[cons_shared_bat_ub[(w, j)].name] = cont

    for k in cons_shared_bat_ub: model.addConstraint(cons_shared_bat_ub[k])

    ## Private Battery lower bound
    cons_shared_bat_lb = {(w, j) :
    plp.LpConstraint(
                 e=-plp.lpSum(shared_ch_vars[(w, t)] - shared_dis_vars[(w,t)] for t in range(j +
                 1)),
                 sense=plp.LpConstraintLE,
                 rhs=batinfo['init'],
                 name="cons_shared_bat_low_{0}_{1}".format(w, j))
           for j in set_T for w in set_W}

    for k in cons_shared_bat_lb: model.addConstraint(cons_shared_bat_lb[k])

    ### Contributions
    for j in set_T:
        for w in set_W:
            cont = np.zeros(N)
            contributions[cons_shared_bat_lb[(w, j)].name] = cont


    ## Public Battery ramp up

    cons_shared_bnd_ub = {(w, t) :
        plp.LpConstraint(
                     e=shared_ch_vars[(w, t)]- batinfo['ram'] * batsize,
                     sense=plp.LpConstraintLE,
                     rhs=0,
                     name="cons_shared_bnd_up_{0}_{1}".format(w, t))
               for t in set_T for w in set_W}

    for k in cons_shared_bnd_ub: model.addConstraint(cons_shared_bnd_ub[k])

    ### Contributions
    for j in set_T:
        for w in set_W:
            cont = np.zeros(N)
            contributions[cons_shared_bnd_ub[(w, j)].name] = cont

    ## Public Battery lower bound
    cons_shared_bnd_lb = {(w, t) :
    plp.LpConstraint(
                 e=shared_dis_vars[(w, t)] - batinfo['ram'] * batsize,
                 sense=plp.LpConstraintLE,
                 rhs=0,
                 name="cons_shared_bnd_low_{0}_{1}".format(w, t))
           for t in set_T for w in set_W}

    for k in cons_shared_bnd_lb: model.addConstraint(cons_shared_bnd_lb[k])

    ### Contributions
    for j in set_T:
        for w in set_W:
            cont = np.zeros(N)
            contributions[cons_shared_bnd_lb[(w, j)].name] = cont

    ####### End public batteries

    ####### Begin fix pv and battery

    ## Fixing battery 
    if batfix is not None:
        cons_fix_bat = plp.LpConstraint(
                     e=batsize,
                     sense=plp.LpConstraintEQ,
                     rhs=batfix,
                     name="batfix")

        for k in cons_fix_bat: model.addConstraint(cons_fix_bat)

        ### Contributions
        cont = np.zeros(N)
        contributions[cons_fix_bat.name] = cont

    ## Fixing PV
    if pvfix is not None:
        cons_fix_pv = plp.LpConstraint(
                     e=pvsize,
                     sense=plp.LpConstraintEQ,
                     rhs=pvfix,
                     name="pvfix")

        for k in cons_fix_pv: model.addConstraint(cons_fix_pv)

        ### Contributions
        cont = np.zeros(N)
        contributions[cons_fix_pv.name] = cont


    #######

    ####### Net energy equations 

    ### On side of the equality

    SEC = 1 / batinfo['ec']
    SED = batinfo['ed']


    cons_z = {(w, t):  
        plp.LpConstraint(
                     e=plp.lpSum(
                        zp_vars[(w,t)] - zn_vars[(w, t)]  
                        - plp.lpSum( ch_vars[(w, n, t)] * (1 / player_list[n]._ec) -
                        dis_vars[(w, n, t)] * player_list[n]._ed for n in set_N)
                        - (shared_ch_vars[(w, t)] * SEC - shared_dis_vars[(w, t)] * SED)
                        - pvsize * pvinfo['data'][(w, t)]
                     ),
                     sense=plp.LpConstraintLE,
                     rhs=sum(player_list[n]._x[w, t] for n in set_N),
                     name="cons_z_{0}_{1}".format(w, t))
                     for t in set_T for w in set_W}
    for k in cons_z: model.addConstraint(cons_z[k])

    ### Contributions
    for w in set_W:
        for j in set_T:
            cont = np.zeros(N)
            for n2 in set_N:
                cont[n2] = player_list[n2]._x[w, j]
            contributions[cons_z[(w, j)].name] = cont

    #### The other side

    cons_zo = {(w, t): 
        plp.LpConstraint(
                     e=-plp.lpSum(
                        zp_vars[(w,t)] - zn_vars[(w, t)]  
                        - plp.lpSum( ch_vars[(w, n, t)] * (1 / player_list[n]._ec) -
                        dis_vars[(w, n, t)] * player_list[n]._ed for n in set_N)
                        - (shared_ch_vars[(w, t)] * SEC - shared_dis_vars[(w, t)] * SED)
                        - pvsize * pvinfo['data'][(w, t)]
                     ),
                     sense=plp.LpConstraintLE,
                     rhs=-sum(player_list[n]._x[w, t] for n in set_N),
                     name="cons_zo_{0}_{1}".format(w, t))
                     for t in set_T for w in set_W}
        
    for k in cons_zo: model.addConstraint(cons_zo[k])


    for w in set_W:
        for j in set_T:
            cont = np.zeros(N)
            for n2 in set_N:
                cont[n2] = - player_list[n2]._x[w, j]
            contributions[cons_zo[(w, j)].name] = cont

    cons = [cons_bat_ub, cons_bat_lb, cons_bnd_ub, cons_bnd_lb,
        cons_shared_bat_ub, cons_shared_bat_lb, cons_shared_bnd_ub, cons_shared_bnd_lb,
        cons_z, cons_zo]

    objective = plp.lpSum(
         - batinfo['cost'] * batsize - pvinfo['cost'] * pvsize
        + plp.lpSum( -zp_vars[(w, t)] * buying_price[t] * prob[w] + zn_vars[(w, t)] *
        selling_price[t] * prob[w]
        for t in set_T for w in set_W)
        )

    model.sense = plp.LpMaximize
    model.setObjective(objective)


    model.solve(plp.PULP_CBC_CMD(msg=False))

    opt_val = objective.value()

    # df = pd.DataFrame(ch_vars.keys())
    # df['ch'] = df.apply(lambda x: ch_vars[(x[0], x[1])].varValue, axis=1)
    # df['dis'] = df.apply(lambda x: dis_vars[(x[0], x[1])].varValue, axis=1)
    # df.columns = ['n', 't', 'ch', 'dis']
    # df_cd = pd.pivot_table(df, index='n', columns='t', values=['ch', 'dis']).round(4)

    # df = pd.DataFrame(zp_vars.keys())
    # df['zp'] = df.apply(lambda x: zp_vars[x[0]].varValue, axis=1)
    # df['zn'] = df.apply(lambda x: zn_vars[x[0]].varValue, axis=1)
    # df.columns = ['t', 'zp', 'zn']
    # df_z = df.copy()

    return model, vars_, cons, contributions


def extract_core_payment(game):

    PL = game._player_list
    pb = game._buying_price
    ps = game._selling_price
    T = len(ps)
    N = len(PL)
    m = game.solve()
    
    cons = m.constraints
    
    payoff = np.zeros(N)
    for n, pl in enumerate(PL):
        pay = 0

        for t in range(T):
            du = cons[f"cons_bnd_up_{n}_{t}"].pi
            pay += du * pl._ram

            du = cons[f"cons_bnd_low_{n}_{t}"].pi
            pay += du * pl._ram

            du = cons[f"cons_bat_up_{n}_{t}"].pi
            pay += du * (pl._sm - pl._s0)

            du = cons[f"cons_bat_low_{n}_{t}"].pi
            pay += du * (pl._s0)

            du = cons[f"cons_z_{t}"].pi
            pay += du * pl._x[t]
            
            du = cons[f"cons_zo_{t}"].pi
            pay += - du * pl._x[t]
            
        payoff[n] = pay

    return payoff
            

def to_matrix_form(g):

    PL = g._player_list
    pb = g._buying_price
    ps = g._selling_price
    T = len(ps)
    N = len(PL)
    m = g.solve()
    cons = [v for lc in g._res[2] for (k, v) in lc.items()]
    n_var = len(m.variables())
    n_con = len(m.constraints)
    ck = [k for k in m.constraints]

    A = np.zeros((n_con, n_var))

    varn = [f'zp_{i}' for i in range(T)]
    varn += [f'zn_{i}' for i in range(T)]
    for n in range(N):
        varn += [f'ch_{n}_{t}' for t in range(T)]
        varn += [f'dis_{n}_{t}' for t in range(T)]

    b = np.zeros(n_con)
    for i, con_ in enumerate(cons):
        
        dcon = dict((k.name, v) for (k, v) in con_.items())
        tmp = [dcon.get(var, 0) for var in varn]
        A[i, :] = np.array(tmp) 
        b[i] = con_.getUb()

    c = np.hstack([
        -pb,
        ps,
        np.zeros(2 * N * T)
    ])

    return A, b, c



def extract_player(n, A, T): return A[:, 2 * T * (n+1): 2 * T * (n + 2)]

def extract_common(A, T): return A[:, :2 * T]

def build_proyection_player(n, game):

    A = game.A
    ps = game._selling_price
    pb = game._buying_price
    T = game.T
    N = game.N

    #A, _, _ = to_matrix_form(game)

    pl_A = extract_player(n, A, T).T.copy()
    N_1 = pl_A.shape[0]
    cm_A = extract_common(A, T).T.copy()
    N_2 = cm_A.shape[0]

    tmp = np.vstack([pl_A, cm_A])
    ma = np.where(np.abs(tmp).sum(axis=0) != 0)[0]
    tmp = tmp[:, ma]
    A_pl = np.vstack([tmp, np.eye(len(ma))])
    b = np.hstack([np.zeros(N_1), -pb, ps, np.zeros(len(ma)) ]) 

    return A_pl, b, ma
        

