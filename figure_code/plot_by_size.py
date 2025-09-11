
#%%
 
import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
from scipy.stats import pearsonr
pysta.reload()
from pysta import basedir
ext = ".pdf"

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% load data

data = pickle.load(open(f"{basedir}/data/comparisons/performance_and_decoding_by_size.pickle", "rb"))
sizes, perfs, decoding = data["sizes"], np.array(data["perfs"]), np.array(data["decoding"])
cols = [plt.get_cmap("viridis")(s/np.amax(sizes)) for s in sizes]

#%%

ymin, ymax = 0.5, 0.95

plt.figure(figsize = (1.8, 1.5))
xs = np.arange(decoding.shape[-1])+1
for isize, size in enumerate(sizes):
    plt.plot(decoding[isize, 0, :], label = f"N={size}", color = cols[isize])
plt.xlabel("time in future")
plt.ylabel("decoding accuracy")
plt.ylim(ymin, ymax)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/acc_vs_time{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

plt.figure(figsize = (1.5, 1.5))
plt.scatter(sizes, decoding[:, 0, :].mean(-1), c = cols)
plt.xlabel("network size")
plt.ylabel("mean decoding")
plt.ylim(ymin, ymax)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/acc_vs_size{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

plt.figure(figsize = (1.5, 1.5))
plt.scatter(sizes, perfs.mean(-1), c = cols)
plt.xlabel("network size")
plt.ylabel("performance")
plt.ylim(ymin, ymax)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/perf_vs_size{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

plt.figure(figsize = (1.5, 1.5))
plt.scatter(decoding[:, 0, :].mean(-1), perfs.mean(-1), c = cols)
plt.xlabel("mean decoding")
plt.ylabel("performance")
plt.ylim(0.6, ymax)
plt.xlim(0.68, 0.9)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/acc_vs_perf{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%%

