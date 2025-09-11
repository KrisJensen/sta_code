

#%%

import numpy as np
import torch
import pysta
pysta.reload()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import os
from pysta import basedir
ext = ".pdf"

#%% load the example maze parameters

from pysta.utils import schematic_walls as walls
adjacency, neighbors = pysta.maze_utils.compute_adjacency(walls)
dists = pysta.maze_utils.compute_shortest_dists(adjacency)
N, L = walls.shape[0], int(np.sqrt(walls.shape[0]))


# %% now plot example environment with and without firing rates

locs = [(0,1), (1,1), (1,0), (2,0)]
loc_inds = np.array([pysta.maze_utils.loc_to_index(loc, L = L) for loc in locs])
dists_to_locs = dists[loc_inds, :]
acts = np.exp(-1.7*dists_to_locs**2)


pysta.reload()
edgecolors = [[1,1,1,0] for _ in range(acts.size)]

sta_sequence_inds = [1, 13, 21, 33]
for iind, ind in enumerate(sta_sequence_inds):
    edgecolors[ind] = sequence_colors[iind]

ax = pysta.plot_utils.plot_perspective_attractor(walls, acts, cmap = "YlOrRd", plot_proj = False, figsize = (5.1, 2.1), aspect = (1,1,2.5), view_init = (-25, -10, -90), plot_subs = True, show = False, edgecolors = edgecolors)
plt.savefig(f"{basedir}/figures/schematics/sta_cells{ext}", bbox_inches = mpl.transforms.Bbox([[1.8,0.55], [3.45,1.54]]), transparent = True)
plt.show()
plt.close()


#%% now plot spikes over time

ys = np.arange(4)
xs = np.zeros(len(ys))+0.5

plt.figure(figsize = (0.6,1.2))
for i in range(len(ys)):
    plt.plot([xs[i], xs[i]], [-ys[i]-0.33, -ys[i]+0.33], color = sequence_colors[i], lw = 3)

plt.xlim(0, 1)
plt.ylim(-3.8, 0.8)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("time")
plt.ylabel("neuron")
plt.gca().spines[['left', 'bottom']].set_linewidth(1.5)
plt.savefig(f"{basedir}/figures/schematics/raster{ext}", bbox_inches = "tight", transparent = True)
plt.show()

# %% now plot ring attractor

styleA = "Simple, tail_width=0.5, head_width=3.5, head_length=4"
styleB = "|-|, widthB=3.5, angleA=0, widthA=0, angleB=0"

thetas = np.linspace(0.5*np.pi, 2.5*np.pi, 17)[:-1]
cell_locs = np.array([np.cos(thetas), np.sin(thetas)]).T

arrows = []
loc0, theta0 = cell_locs[0], thetas[0]
for iloc, loc in enumerate(cell_locs[1:]):
    dtheta = thetas[iloc+1] - theta0
    dtheta = dtheta if dtheta <= np.pi else dtheta - 2*np.pi
    ex = np.abs(dtheta) < np.pi/2
    style = styleA if ex else styleB
    kw = dict(arrowstyle=style, color=plt.get_cmap("coolwarm")(1-np.abs(dtheta)/np.pi))
    constyle = "arc3,rad="+str(-np.sign(dtheta)*0.7*(1-np.abs(dtheta/np.pi)))
    print(loc, dtheta, constyle)
    a = patches.FancyArrowPatch(0.91*loc0, 0.91*loc,
                             connectionstyle=constyle, **kw, lw = 1.0 if ex else 1.7)

    arrows.append(a)

plt.figure(figsize = (1.4,1.4))
plt.scatter(cell_locs[:, 0], cell_locs[:, 1], marker = "o",
            color = np.ones(3)*0.7, s = 70, edgecolors = np.ones(3)*0.5)
for a in arrows:
    plt.gca().add_patch(a)
plt.gca().axis("off")
plt.xlim(-1.15, 1.15)
plt.ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig(f"{basedir}/figures/schematics/bg_hd_connections{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

thetas_act = np.linspace(0.5*np.pi, 2.5*np.pi, 30)[:-1]
xs = np.arange(len(thetas_act))
ring_acts = [np.cos(theta - thetas_act[9]) for theta in thetas_act]
fig = plt.figure(figsize = (1.5, 1.0))
ax = plt.gca()
for ix, act in enumerate(ring_acts):
    ax.bar(xs[ix], [act], color = plt.get_cmap("YlOrRd")(act))

plt.xticks([])
plt.yticks([])
plt.ylabel("activity")
plt.ylim(0, 1.3)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.gca().spines[['left', 'bottom']].set_linewidth(1.5)

plt.tight_layout()
plt.xlabel("preferred angle")
plt.savefig(f"{basedir}/figures/schematics/bg_hd_activity{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



# %% now plot an example RNN

cell_locs = np.array([
    [3,8],
    [5.5,2],
    [8,4],
    [2,4],
    [3.2,2],
    [7.8,6],
    [6,8],
    [4.7,5.2]
])
cell_locs = cell_locs[np.argsort(cell_locs[:, 0])]

minv, maxv = np.amin(cell_locs, axis = 0), np.amax(cell_locs, axis = 0)

cell_locs = (cell_locs - minv[None, :])/(maxv-minv)[None, :]*2-1 # shift to origin

plt.figure(figsize = (1.5,1.5))
plt.scatter(cell_locs[:,0], cell_locs[:,1], s = 160, color = np.ones(3)*0.7, edgecolor = np.ones(3)*0.5)


cons = [
    (0, 3, -0.2, "+"),
    (2,6, -0.1, "-"),
    (1,4, 0.2, "+"),
    (7,1,0.1,"-"),
    (5,4,-0.1,"+"),
    (1,0,0.2,"-"),
    (4,7,0.2,"+"),
    (6,5,0.2,"+"),
    (3,5,-0.2,"-")
]

for con in cons:
    loc0, loc1, rad = np.array(cell_locs[con[0]]), np.array(cell_locs[con[1]]), con[2]
    dloc = loc1 - loc0
    dloc = dloc / np.sqrt(np.sum(dloc**2)) # normalize
    ex = True if con[3] == "+" else False
    style = styleA if ex else styleB
    
    kw = dict(arrowstyle=style, color=plt.get_cmap("coolwarm")(0.8 if ex else 0.2))
    
    constyle = "arc3,rad="+str(rad)
    a = patches.FancyArrowPatch(loc0+0.18*dloc, loc1-0.15*dloc,
                             connectionstyle=constyle, **kw, lw = 1.3 if ex else 1.7)
    plt.gca().add_patch(a)


plt.xlim(-1.2, 1.2)
plt.ylim(-1.2,1.2)
plt.gca().axis("off")
plt.savefig(f"{basedir}/figures/schematics/bg_rnn{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()
    
# %%
