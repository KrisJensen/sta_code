"""Code for plotting the RNN inputs"""

#%% load libraries
import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
import os
import matplotlib as mpl
from pysta import basedir
from scipy.spatial.transform import Rotation
ext = ".pdf"
basefigdir = f"{basedir}/figures/rnn_inputs/"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8


# %% load data
ex_seed = 31
example_model = f"sta_MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model{ex_seed}"
connectivity_data = pickle.load(open(f"{basedir}/data/rnn_analyses/{example_model}_connectivity_data.pickle", "rb"))

env = connectivity_data["rnn"].env
walls = env.walls[0]
rs1 = env.rews[0][1]
rs2 = env.rews[0][2]
rs3 = env.rews[0][3]
rs4 = rs1*0.0
adj = env.adjacency[0].detach().numpy()

vmap = np.zeros((3, env.num_locs))
vmap += np.random.uniform(0.15,0.35, vmap.shape)


#%% scaffold and 'empty' STA

plt.figure(figsize = (2.4,2.4))
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adj, ax = ax)
ax.axis("off")
plt.savefig(f"{basefigdir}schematic_maze_scaffold{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



#%% future rewards

vmin, vmax, cmap = -1.2,2.2,"Greens"
for ir, rs in enumerate([rs1, rs2, rs3, rs4]):
    plt.figure(figsize = (1.0, 1.0))
    ax = plt.gca()
    pysta.plot_utils.plot_flat_frame(walls, ax = ax, filename = None, vmap = rs, cmap = cmap, vmin = vmin, vmax = vmax, show = False, lw = 3.0)
    plt.savefig(f"{basefigdir}schematic_rewards{ir}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

# now plot policy

pi = np.random.uniform(0,0.1, 16)
pi[1] = 0.15
pi[2] = 0.25
pi[3] = 0.5
pi = pi / pi.sum()

plt.figure(figsize = (2.4,2.4))
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adj, ax = ax, vmap = pi)
ax.axis("off")
plt.savefig(f"{basefigdir}policy{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

# %%
