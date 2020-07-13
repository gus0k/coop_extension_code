param=$1
cat "sim/params_$param.csv" | parallel -C, --header : "venv/bin/python sim/sim_compare_graphs.py {N} {T} {G} {S}" &
