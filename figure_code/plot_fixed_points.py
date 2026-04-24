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
bbox_inches = mpl.transforms.Bbox([[0.92,0.8], [2.66,1.42]])
aspect, figsize = (1,1,5), (3.5,2.2)
fp_cols = [np.array([31, 100, 200])/256, np.array([31, 140, 160])/256]

#%%

results = pickle.load(open(f"{basedir}/data/examples/fixed_points.p", "rb"))

#%% plot a scaffold of the open arena

walls_empty, walls_twopaths, walls_goodbad = [results[key]["walls"][0] for key in ["empty", "twopaths", "goodbad"]]

for iw, walls in enumerate([walls_empty, walls_twopaths, walls_goodbad]):
    adjacency = pysta.maze_utils.compute_adjacency(walls)[0].numpy()

    plt.figure(figsize = (1.6, 1.6))
    ax = plt.gca()
    pysta.plot_utils.plot_maze_scaffold(adjacency, ax = ax, s = 500, lw = 6.5)
    ax.axis("off")
    plt.savefig(f"{basedir}/figures/fixed_points/walls{iw}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

#%% plot example representations throughout the empty trial and the twopath trial
rs_empty = results["empty"]["rs"]
rs_twopaths = results["twopaths"]["example_convergence"]

for iw, rs in enumerate([rs_empty, rs_twopaths]):
    inds = [[0, 50, 150, 1000], [0, 150, 300, 1000]][iw]
    walls = [walls_empty, walls_twopaths][iw]

    to_plot = [rs[ind] for ind in inds]
    for ir, r in enumerate(to_plot):
        ax = pysta.plot_utils.plot_perspective_attractor(walls, r[0], plot_proj = True, cmap = "YlOrRd", figsize = figsize, aspect = aspect, view_init = (-30,-10,-90),
        filename = f"{basedir}/figures/fixed_points/con_{['empty', 'twopaths'][iw]}_{ir}.pdf",  show = True, bbox_inches = bbox_inches, vmin = -0.1, vmax = 1.2)

#%% plot fixed points for the twopaths and goodbad trials

fixed_points = [[results["twopaths"]["norew"]], results["twopaths"]["fixed_points"], results["goodbad"]["fixed_points"]]
labels = ["norew", "twopaths", "goodbad"]

for i_f, fixed in enumerate(fixed_points):
    walls = [walls_twopaths, walls_twopaths, walls_goodbad][i_f]
    for ir, r in enumerate(fixed):
        print(r.shape)
        ax = pysta.plot_utils.plot_perspective_attractor(walls, r, plot_proj = True, cmap = "YlOrRd", figsize = figsize, aspect = aspect, view_init = (-30,-10,-90),
        filename = f"{basedir}/figures/fixed_points/fixed_{labels[i_f]}_{ir}.pdf",  show = True, bbox_inches = bbox_inches, vmin = -0.1, vmax = 1.2)



# %% plot fixed point distribution

all_fits = [results["twopaths"]["fits"]] + results["goodbad"]["fits"]
#all_fits[0] = all_fits[0][:, np.array([1,0])]
xs = [0, 1]
for ifits, fits in enumerate(all_fits):
    ms = fits.mean(0)
    if ifits == 0:
        plt.figure(figsize = (1.0, 1.0))
    else:
        plt.figure(figsize = (0.85, 1.0))
    for i in range(2):
        plt.bar(xs[i:i+1], ms[i:i+1], color = fp_cols[i])

    plt.xticks(xs, ["FP 1", "FP 2"])
    plt.ylabel("frequency", labelpad = -6)
    plt.gca().tick_params(axis='x', which='major', pad=2)

    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.axhline(0.5, color = "grey")
    plt.ylim(0.0, 1.0)
    plt.yticks([0, 1])
    plt.savefig(f"{basedir}/figures/fixed_points/fits_{ifits}.pdf", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

# %%
