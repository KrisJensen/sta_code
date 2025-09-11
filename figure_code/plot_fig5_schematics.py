
#%% load libraries

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
pysta.reload()
import os
import matplotlib as mpl
from pysta import basedir



# %% load data

example_model = "sta_MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model22"
connectivity_data = pickle.load(open(f"{basedir}/data/rnn_analyses/{example_model}_connectivity_data.pickle", "rb"))

env = connectivity_data["rnn"].env
walls = env.walls[0]
rs1 = env.rews[0][1]
adj = env.adjacency[0].detach().numpy()
stim_col = (0.35, 0.65, 0.2)

vmap = np.zeros((3, env.num_locs))
vmap += np.random.uniform(0.15,0.35, vmap.shape)

#%% plot a colorbar

plt.figure(figsize = (0.6,4))
plt.imshow(np.linspace(+1, -1, 101)[:, None], vmin = -1, vmax = +1, aspect = "auto", cmap = "coolwarm")
plt.xticks([])
plt.yticks([])
plt.savefig(f"{basedir}/figures/rnn_connectivity/conn_cbar{ext}", bbox_inches = "tight", transparent = True)
plt.show()

#%% example 'connection'

ind0, ind1 = 6, 7
act_cols = [[np.ones(3)*0.92 for _ in range(env.num_locs)] for _ in range(3)]
act_cols[0][ind0] = stim_col
act_cols[1][ind1] = (0.95, 0.1, 0.95)

bbox = mpl.transforms.Bbox([[2.1,1.0], [5.15,2.95]])
pysta.plot_utils.plot_perspective_attractor(walls, vmap[:2],  act_cols = act_cols, vmin = 0, vmax = 1, cmap = "coolwarm",
                                            lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,2.0), view_init = (-35,-10,-90),
                                            bbox_inches = bbox, transparent = True, filename = f"{basefigdir}schematic_coordinates{ext}")


#%% schematic weights


                
adj_inds = np.where(adj[ind0, :] > 0)[0]
for aind in adj_inds:
    act_cols[1][aind] = plt.get_cmap("coolwarm")(0.95)

bbox = mpl.transforms.Bbox([[2.1,1.0], [5.15,2.95]])
pysta.plot_utils.plot_perspective_attractor(walls, vmap[:2],  act_cols = act_cols, vmin = 0, vmax = 1, cmap = "coolwarm",
                                            lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,2.0), view_init = (-35,-10,-90),
                                            bbox_inches = bbox, transparent = True, filename = f"{basefigdir}schematic_weights{ext}")
                


#%% scaffold and 'empty' STA

plt.figure(figsize = (2.4,2.4))
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adj, ax = ax)
ax.axis("off")
plt.scatter([1,], [2,], marker = ".", color = stim_col, s = 1600, zorder = 200)
plt.savefig(f"{basefigdir}schematic_maze_scaffold{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

pysta.plot_utils.plot_perspective_attractor(walls, vmap, vmin = 0, vmax = 1,
                                            lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,2.0), view_init = (-30,-10,-90),
                                            bbox_inches = bbox, transparent = True, filename = f"{basefigdir}schematic_activity{ext}")
                


#%% future rewards

vmin, vmax = -2.3, 1.15
plt.figure(figsize = (2.0,2.0))
ax = plt.gca()
pysta.plot_utils.plot_flat_frame(walls, ax = ax, filename = None, vmap = rs1, cmap = "viridis", vmin = vmin, vmax = vmax, show = False)
plt.scatter([2,], [1,], marker = ".", color = stim_col, s = 1600, zorder = 200)
plt.savefig(f"{basefigdir}schematic_rewards{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



# %%
