
#%%

import pysta
import os
from pysta import basedir

os.makedirs(f"{basedir}/data/rnn_analyses", exist_ok = True)
os.makedirs(f"{basedir}/data/comparisons", exist_ok = True)

os.makedirs(f"{basedir}/figures", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_decoding", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_connectivity", exist_ok = True)
os.makedirs(f"{basedir}/figures/schematics", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_generalisation", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_behaviour", exist_ok = True)
os.makedirs(f"{basedir}/figures/simple_tasks", exist_ok = True)
os.makedirs(f"{basedir}/figures/size", exist_ok = True)
os.makedirs(f"{basedir}/figures/changing_maze_rnn", exist_ok = True)
os.makedirs(f"{basedir}/figures/attractor", exist_ok = True)

os.makedirs(f"{basedir}/models", exist_ok = True)

#%%
