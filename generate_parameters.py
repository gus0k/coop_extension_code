N = 40
W = 40
T = 48
D = 20
cant_bats = 0
cant_solar = 0
real_data = 10
seed = 1234
cost_solar = 80
bpt = 2
bp1 = 20
bp2 = 15
bp3 = 0
sp = 5
size_bat = 13
init_bat = 0
ram_bat = 2.5
ec_bat = 0.95
ed_bat = 0.95
cost_bat = 170
max_solar = -0.75
name = 'test3'

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
STARTING = [10, 15, 20, 25, 30, 35]

for seed in SEEDS:
    for real_data in STARTING:
        parameters = [N, W, T, D, cant_bats, cant_solar, real_data, seed, cost_solar, bpt, bp1, bp2, bp3, sp, size_bat, init_bat, ram_bat, ec_bat, ed_bat, cost_bat, max_solar, name]
        string = ','.join(map(str, parameters))
        print(string)
