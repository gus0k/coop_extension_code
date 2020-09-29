import os
import socket
from os.path import expanduser

hostname = socket.gethostname()

home = expanduser("~")
SIMULATION_PARAMETERS = home + '/Simulations/coop_extension'

if not os.path.isdir(SIMULATION_PARAMETERS):
    os.makedirs(SIMULATION_PARAMETERS)
if hostname == 'x220':
    DATAPATH = home + '/github/coop_extension_code/load_profiles/'
    CPLEX_PATH = '/home/diego/cplex/cplex/bin/x86-64_linux/cplex'
elif hostname == 'lame23':
    DATAPATH = home + '/coop_extension_code/coop_extension_code/load_profiles/'
#    CPLEX_PATH = '/home/infres/dkiedanski/Cplex/cplex/bin/x86-64_linux/cplex'
#    CPLEX_PATH=None
    CPLEX_PATH = '/home/infres/dkiedanski/cplex2/cplex/bin/x86-64_linux/cplex'
elif hostname == 'diegoxps15':
    DATAPATH = home + '/github/coop_extension_code/load_profiles/'
    CPLEX_PATH = '/home/diego/cplex/cplex/bin/x86-64_linux/cplex' 

DATA = DATAPATH + 'home_data_2012-13.csv'
DATA_SOLAR = DATAPATH + 'home_data_2012-13_rand_03.csv'
DATA_FORCAST = DATAPATH + 'home_data_2012-13_forcast.csv'
DATA_SOLAR_FORCAST = DATAPATH + 'home_data_2012-13_rand_03_forcast.csv'




