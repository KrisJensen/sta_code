
#%% load libraries

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import matplotlib as mpl
import torch
from scipy.sparse.csgraph import dijkstra
pysta.reload()
from pysta.utils import compute_model_support
from pysta import basedir
from scipy.ndimage import gaussian_filter1d
ext = ".pdf"

#%% generate little schematic of relative vs absolute

locs = np.arange(6)
neurals = np.arange(4)
sequence_colors = [plt.get_cmap("viridis")(iind / (len(neurals)-0.5)) for iind in range(len(neurals))][::-1]
for i in range(2):
    plt.figure(figsize = (1.2, 0.9))
    for t in neurals:
        perf = np.zeros(len(locs))
        if i == 0:
            perf[locs == t+2] = 1.0
        else:
            perf[locs == 3] = 0.9**t
        perf = perf + (1-perf.sum())/len(perf)
        plt.plot(locs, perf, lw = 2, color = sequence_colors[int(t)])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{pysta.basedir}/figures/schematics/decoding_support_schematic{i}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


# %% plot some example representations

from pysta.utils import schematic_walls as walls
num_locs = walls.shape[0]
L = int(np.sqrt(num_locs))
num_mod = 4


adjacency = pysta.maze_utils.compute_adjacency(walls)[0].numpy()

all_dists = dijkstra(pysta.maze_utils.compute_adjacency(walls)[0], directed=False, unweighted = True)

all_locs = [[(1,2), (2,2), (2,1), (1,1), (1,0)],
            [(2,2), (2,1), (1,1), (1,0)],
            [(2,0), (1,0), (1,1), (0,1)]] # trajectories
num_mod = 4

for ipath, locs in enumerate(all_locs):
    num_mod = min(num_mod, len(locs))
    locs = locs[:num_mod]
    
    path_inds = [pysta.maze_utils.loc_to_index(loc, L) for loc in locs if len(loc) == 2]
    
    vmap = np.zeros((num_mod, num_locs))
    for z in range(len(locs)):
        if z < len(path_inds):
            vmap[z, :] = np.exp(-1.7*all_dists[path_inds[z], :]**2)
        else:
            vmap[z, :] = np.exp(-1.7*np.amin(all_dists[adjacency[path_inds[z-1], :].astype(bool), :], 0)**2)/adjacency[path_inds[z-1], :].sum()

    edgecolors = [[1,1,1,0] for _ in range(vmap.size)]
    #edgecolors[9*2 + 4] = np.array([240,106,167])/255
            
    pysta.reload()
    ax = pysta.plot_utils.plot_perspective_attractor(walls, vmap, edgecolors = edgecolors, filename = None, plot_proj = False, cmap = "YlOrRd", vmin = -0.15, vmax = 1.05, figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90))
    plt.savefig(f"{basedir}/figures/rnn_decoding/pfc_activity{ipath}{ext}", bbox_inches = mpl.transforms.Bbox([[0.90,0.55], [2.7,1.65]]), transparent = True)
    plt.show()
    plt.close()
    
    
    
#%% plot scaffolds with trajectories on top
istarts = [0, 1, 0]
for ilocs, locs in enumerate(all_locs):
    locs = np.array(locs).T.astype(float)

    plt.figure(figsize = (1.43,1.43))
    ax = plt.gca()
    pysta.plot_utils.plot_maze_scaffold(adjacency, ax = ax, s = 450, lw = 7)

    smooth_locs = gaussian_filter1d(np.repeat(locs, 6, axis = -1), 2, axis = -1, mode = "nearest").T
    chunks = np.array_split(np.arange(len(smooth_locs)), locs.shape[-1])
    cols = [plt.get_cmap("viridis")(0.7*(i)/5 + 0.40) for i in range(5)][::-1][istarts[ilocs]:]
    for ichunk, chunk in enumerate(chunks):
        if ichunk < len(chunks)-1:
            chunk = np.concatenate([chunk, chunks[ichunk+1][:1]])
        ax.plot(smooth_locs[chunk, 0], smooth_locs[chunk, 1], color = cols[ichunk], lw = 5)

    #ax.plot(smooth_locs[:, 0], smooth_locs[:,1])
    ax.axis("off")
    plt.savefig(f"{basedir}/figures/rnn_decoding/maze_scaffold{ilocs}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

 
 #%% and plot an example 'space-by-time' reward function

np.random.seed(0)
torch.manual_seed(0)

env = pysta.envs.MazeEnv(side_length = 3, max_steps = 3, changing_trial_rew = True, batch_size = 1)
env.walls[0, ...] = torch.tensor(walls[...])

path = [1,4,3,6]
optimal = False
while not optimal:
    #print("resampling")
    env.reset()
    vs = env.vs[0].detach().numpy()
    optimal = True
    for i in range(len(path)-1):
        adjacent_locs = np.where(adjacency[path[i]])[0]
        if adjacent_locs[np.argmax(vs[i+1, adjacent_locs])] != path[i+1]:
            optimal = False

vs = env.vs.detach().numpy()[0]
vs = ((vs - vs.mean(-1)[:, None]) / vs.std(-1)[:, None]) # normalize within each subspace

ax = pysta.plot_utils.plot_perspective_attractor(walls, vs, filename = None, plot_proj = False, cmap = "YlOrRd", figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90))
plt.savefig(f"{basedir}/figures/rnn_decoding/value_function{ext}", bbox_inches = mpl.transforms.Bbox([[0.90,0.55], [2.7,1.65]]), transparent = True)
plt.show()
plt.close()


# %%
