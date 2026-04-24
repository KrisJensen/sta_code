"""Code for plotting all panels related to the attrator dynamics analyses"""

#%%
import torch
import matplotlib.pyplot as plt
import pysta
import pickle
import numpy as np
from pysta import basedir
import matplotlib as mpl
pysta.reload()
ext = ".pdf"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%%

results = pickle.load(open(f"{basedir}/data/comparisons/perf_by_length.p", "rb"))
lengths, noises, good = [results[key] for key in ["lengths", "noises", "goods"]]


#%% plot mean performance

means = good.mean(1)
sems = good.std(1)/np.sqrt(good.shape[1])
labels = ["no noise", "low", "medium", "high"]
plt.figure(figsize = (2.7, 1.8))
for i in range(len(means)):
    x, m, s = lengths, means[i], sems[i]
    plt.plot(x, m, label = labels[i])
    plt.fill_between(x, m-s, m+s, alpha = 0.2, edgecolor = [1,1,1,0])

plt.xticks(range(1,15,3))
plt.xlabel("distance to goal", labelpad = 3.5)
plt.ylabel("fraction correct", labelpad = -7)
plt.ylim(0, 1.03)
plt.xlim(lengths[0], lengths[-1])
plt.yticks([0, 1])
plt.legend(loc = "upper center", bbox_to_anchor = (0.25, 0.65), ncol = 1, frameon = False, columnspacing = 0.9, handlelength = 1.1, handletextpad = 0.5)
plt.gca().spines[['right', 'top']].set_visible(False)

plt.savefig(f"{pysta.basedir}/figures/perf_by_length/quantification{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% plot long mean performance

long_results = pickle.load(open(f"{basedir}/data/comparisons/perf_by_length_long.p", "rb"))
long_lengths, long_good = [long_results[key] for key in ["lengths", "goods"]]
means = long_good.mean(0)
sems = long_good.std(0)/np.sqrt(long_good.shape[0])
plt.figure(figsize = (2.7, 1.8))

x, m, s = long_lengths, means, sems
plt.plot(x, m)
plt.fill_between(x, m-s, m+s, alpha = 0.2, edgecolor = [1,1,1,0])

plt.xticks(range(1,31,7))
plt.xlabel("distance to goal", labelpad = 3.5)
plt.ylabel("fraction correct", labelpad = -7)
plt.ylim(0, 1.03)
plt.xlim(long_lengths[0], long_lengths[-1])
plt.yticks([0, 1])
plt.gca().spines[['right', 'top']].set_visible(False)

plt.savefig(f"{pysta.basedir}/figures/perf_by_length/quantification_long{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% also plot example arena
np.random.seed(2)
dists = np.zeros(1)
while dists.max() < lengths[-1]:
    env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_rew = False, batch_size = len(lengths), dynamic_rew = False, max_steps = lengths[-1]+1, side_length = 6)
    dists = pysta.maze_utils.compute_shortest_dists(env.adjacency[0])
#goal = np.random.choice(np.where(dists >= lengths[-1])[0])

adjacency = env.adjacency.numpy()[0]

plt.figure(figsize = (2.1, 2.1))
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adjacency, ax = ax, s = 190, lw = 4)
ax.axis("off")
plt.savefig(f"{basedir}/figures/perf_by_length/scaffold{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


# %% plot hierarchy

results = pickle.load(open(f"{basedir}/data/comparisons/hierarchy.p", "rb"))
walls, abs_r, big_r, start, goal = [results[k] for k in ["walls", "abs_r", "big_r", "start", "goal"]]


# plot arena

ax = pysta.plot_utils.plot_flat_frame(big_walls, figsize = (3.1,3.1))
vmap = np.ones((16*16, 4))
cols = [plt.get_cmap("tab20")(i) for i in np.random.choice(20, 16, replace = False)]
for s in range(16):
    vmap[abs2conc[s], :] = cols[s]
plt.imshow(vmap.reshape(16,16,4))
plt.savefig(f"{basedir}/figures/perf_by_length/hierarchy_arena.pdf", bbox_inches = "tight", transparent = True)
plt.show()

#%% plot abstract
aspect = (1, 1, 6.5)
figsize = (17, 6.5)
view_init = (-34,-10,-90)
subs = 7
bbox_inches = mpl.transforms.Bbox([[6.15,2.55], [11.3,4.0]])

plot_kwargs = {"walls": big_env.walls[0], "view_init": view_init, "vmap": abs_r_big[:subs], "loc": None, "goal": None, "aspect": aspect, "figsize": figsize, "bbox_inches": bbox_inches}
pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, filename = f"{basedir}/figures/perf_by_length/big_abs_plan.pdf", show = True, vmin = -0.1, vmax = 1.2)

#%% plot concrete

plot_kwargs["vmap"] = big_r[:subs]
pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, filename = f"{basedir}/figures/perf_by_length/big_plan.pdf", show = True, vmin = -0.1, vmax = 1.2)





# %%
