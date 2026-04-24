

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



#%%

results = pickle.load(open(f"{basedir}/data/rnn_analyses/subspace_variance_explained.pickle", "rb"))

var_exp, ts = results

model_labels = ["WM", "relrew", "STA"]
figsize = (2.8, 1.85)

for imodel, model_results in enumerate(var_exp):

    ncomp = model_results[-1].shape[-1]

    vars = np.array([v[..., :ncomp] for v in model_results]) # subspace_type x seed x time x component

    plot_vars = vars.sum(-1).mean(1) # cumulative var, then avg over seeds
    plot_stds = vars.sum(-1).std(1) # std over seeds

    # plot_vars = np.array([v[..., :ncomp].sum(-1).mean(0) for v in model_results])
    # plot_stds = np.array([v[..., :ncomp].sum(-1).std(0) for v in model_results])

    pca_vars = model_results[0] # seeds by time by component
    prs = pca_vars.sum(-1)**2 / (pca_vars**2).sum(-1)
    print(np.round(prs.mean(0), 1))

    plot_inds = [1,2,0]
    labels = [f"ceiling", "planning\nsubspaces", "execution\nsubspaces"]
    cols = [np.zeros(3)+0.6, plt.get_cmap("tab10")(1), plt.get_cmap("tab10")(0)]

    plt.figure(figsize = figsize)
    #plt.axvline(-0.5, color = "k", ls = "-")
    plt.plot([-0.5, -0.5], [0.0, 0.9], color = "k", ls = "-", zorder = -100)
    for i in plot_inds:
        m, s = plot_vars[i], plot_stds[i]
        plt.plot(ts, m, label = labels[i], color = cols[i])
        plt.fill_between(ts, m-s, m+s, alpha = 0.2, edgecolor = [1,1,1,0], color = cols[i])
    plt.axhline(ncomp/800, color = "k", ls = "--", label = "chance", zorder = -100)
    plt.xticks(ts[::2])
    plt.xlabel("time within trial", labelpad = 3.5)
    plt.ylabel("variance explained", labelpad = -7)
    plt.ylim(0, 1.0)
    plt.xlim(ts[0], ts[-1])
    plt.yticks([0, 1])
    if imodel == 0:
        reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
        h, l = plt.gca().get_legend_handles_labels()
        plt.gca().legend(reorder(h, 2), reorder(l, 2), loc = "upper center", bbox_to_anchor = (0.55, 1.23), ncol = 2, frameon = False, columnspacing = 0.9, handlelength = 1.4, handletextpad = 0.5)
    
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.savefig(f"{pysta.basedir}/figures/rnn_variance/var_{model_labels[imodel]}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

    col_scales = [1.61, 1.51, 1.41, 1.1, 0.8, 0.5][::-1]

    for itype, type_ in enumerate(["PCA", "sub"]):
        plt.figure(figsize = figsize)

        if type_ == "PCA":
            ind = 0
            block_size = 1
            ex_ts = [7,8,9,10,11,12]
            col_scales = np.linspace(0.0, 1.2, 6)[::-1]
        else:
            ind = 2
            block_size = 16
            ex_ts = [9,10,11,12]
            col_scales = [1.41, 1.1, 0.8, 0.5]
        var_type = vars[ind]


        for it, t in enumerate(ex_ts):
            blocks = int(ncomp / block_size)
            var_block = np.array([var_type[..., t, b*block_size : (b+1)*block_size] for b in range(blocks)]).sum(-1) # sum over components in each block

            if type_ == "PCA":
               #var_block = np.log(var_block)
               var_block = np.concatenate([np.zeros((1, 5)), np.cumsum(var_block, 0)])
               None

            #var_block = np.cumsum(var_block, 0)

            m = var_block.mean(-1).T
            s = var_block.std(-1).T
            xs = np.arange(len(m))

            col = np.array(cols[ind]) #np.array(plt.get_cmap("tab10")(i))
            col[:3] *= col_scales[it]
            plt.plot(xs, m, label = f"t={int(ts[t])}" if i == 0 else None, color = col)
            plt.fill_between(xs, m-s, m+s, alpha = 0.2, edgecolor = [1,1,1,0], color = col)

        plt.xlim(xs[0], xs[-1])
        
        if type_ == "PCA":
            plt.ylim(0, 1.0)
            plt.xlabel("principal component", labelpad = 3.5)
            plt.ylabel("cumulative variance", labelpad = 2)
            if imodel == 2:
                plt.legend(loc = "upper center", bbox_to_anchor = (0.68, 0.50), ncol = 3, frameon = False, columnspacing = 0.8, handlelength = 1.1, handletextpad = 0.4)
        
        else:
            plt.ylim(0.0, 0.35)
            plt.xlabel("subspace index", labelpad = 3.5)
            plt.ylabel("variance explained", labelpad = 2)
            if imodel == 2:
                plt.legend(loc = "upper center", bbox_to_anchor = (0.505, 0.22), ncol = 4, frameon = False, columnspacing = 0.8, handlelength = 1.1, handletextpad = 0.4)

        if imodel in [0,1]:
            plt.yticks([])
            plt.ylabel(None)

        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.savefig(f"{pysta.basedir}/figures/rnn_variance/var_by_block_{type_}_{model_labels[imodel]}{ext}", bbox_inches = "tight", transparent = True)
        plt.show()
        plt.close()



# %%
