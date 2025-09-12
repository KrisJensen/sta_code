
#%% load libraries

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
import matplotlib as mpl
from scipy.stats import pearsonr
pysta.reload()
from pysta import basedir

ext = ".pdf"
basefigdir = f"{basedir}/figures/changing_maze_rnn/"

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% load data
seeds = [21,22,23,24,25]
model_names = [f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_changing-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model{seed}" for seed in seeds]
model_names_ref = [name.replace("changing-maze", "constant-maze") for name in model_names]
model_names = [f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_changing-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/old_model{seed}" for seed in [21,23,24,25]]
datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(model_name.split("/")) + "_" for model_name in model_names]
datadirs_ref = [f"{basedir}/data/rnn_analyses/" + "_".join(model_name.split("/")) + "_" for model_name in model_names_ref]


#%% Plot performance as a bar plot

perfs = pickle.load(open(f"{basedir}/data/comparisons/rnn_generalisation.pickle", "rb"))["perfs"]

for ienv, env in enumerate([0, 3]):
    data =  perfs[..., env, np.array([0,3]), 0].mean(1) # performance of static/changing models in static/changing tasks across seeds and repeats, then avg across repeats
    plt.figure(figsize = (1.6,2))

    xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
    jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
    plt.bar(xs, ms, yerr = ss, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
    for idata, datapoints in enumerate(data.T):
        plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)

    plt.xticks(xs, ["fixed\nmaze", "changing\nmaze"])#, rotation = 45, ha = "right")
    plt.axhline(0.2, color = np.ones(3)*0.6)
    plt.ylabel("accuracy", labelpad = -7)
    plt.yticks([0, 1])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{basefigdir}performance{ienv}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

#%% load data on effective connectivity

changing_maze_results = pickle.load(open(f"{datadirs[2]}correlation_results.pickle", "rb")) # load data for example agent
walls = changing_maze_results["wall_configs"] # the walls used in the trials

#%% plot two examples mazes

all_adjs = np.array(changing_maze_results["adjs"]).reshape((-1, 16*16))
all_adjs = 2*all_adjs - 1

adj_cors = (all_adjs[None, ...] * all_adjs[:, None, :]).mean(-1) # (num_walls, num_walls, 16*16)
ind1, ind2 = [arr[0] for arr in np.where(adj_cors == np.amin(adj_cors))]
adj1, adj2 = all_adjs[ind1], all_adjs[ind2]

adj_stds = np.std(all_adjs, axis = 0)
always_same = np.zeros((16, 16))
always_same[(adj_stds == 0).reshape(16, 16)] = np.nan

for iind, ind in enumerate([ind1, ind2]): # for each example
    # plot the maze structure
    pysta.plot_utils.plot_flat_frame(filename = f"{basefigdir}ex_maze{iind}{ext}", vmap = np.zeros((4,4)), vmin = -1, vmax = 1, cmap = "coolwarm", walls = walls[ind], show = True)

    # plot the true adjacency matrix
    plt.figure(figsize = (1.5,1.5))
    plt.imshow(changing_maze_results["adjs"][ind] + always_same, cmap = "coolwarm")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{basefigdir}ex_adj{iind}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
    
    # plot the inferred adjacency matrix
    plt.figure(figsize = (1.5,1.5))
    Weff = changing_maze_results["Weffs"][ind]
    vmin, vmax = np.quantile(Weff, 0.05), np.quantile(Weff, 0.95)
    plt.imshow(Weff + always_same, cmap = "coolwarm", vmin = vmin, vmax = vmax)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{basefigdir}ex_W{iind}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

#%% plot correlation statistics

cols = [np.array(plt.get_cmap("tab10")(0)) for _ in range(2)]
cols[1] = np.zeros(3)+0.6

true_cors, false_cors, false_adj_cors = [changing_maze_results[key] for key in ["true_cors", "false_cors", "false_adj_cors"]]

# print(len(true_cors), len(false_cors), len(false_adj_cors))
# print(np.mean(true_cors), np.mean(false_cors), np.mean(false_adj_cors))

mtrue, strue = np.mean(true_cors), np.std(true_cors)
bins_true = np.linspace(np.amin(true_cors)-1e-5, np.amax(true_cors)+1e-5, 6)
bins_false = np.linspace(np.amin(false_cors)-1e-5, np.amax(false_cors)+1e-5, 12)


plt.figure(figsize = (3.0,2))
countst, _, _ = plt.hist(true_cors, bins = bins_true, label = "same maze", density = True, color = cols[0])
countsf, _, _ = plt.hist(false_cors, bins = bins_false, label = "different maze", density = True, color = cols[1])

ymin, ymax = 0, np.ceil(max(np.amax(countsf), np.amax(countst))/5)*5
#plt.axvline(mtrue, color = "k", lw = 3, label = "same maze")
#plt.fill_between([mtrue-2*strue, mtrue+2*strue], [ymin, ymin], [ymax, ymax], color = "k", alpha = 0.3, linewidth = 0)
plt.ylim(ymin, ymax)
plt.xlabel("correlation of W with A")
plt.ylabel("frequency")
plt.yticks([])
plt.legend(frameon = False)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}adjacency_matrix_correlations{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% now plot avg across mazes

for iref, plot_dirs in enumerate([datadirs, datadirs_ref]):

    all_maze_results = [pickle.load(open(f"{datadir}correlation_results.pickle", "rb")) for datadir in plot_dirs] # save the trial data
    all_true, all_false = [], []
    for maze_results in all_maze_results:
        all_true.append(np.mean(maze_results["true_cors"]))
        all_false.append(np.mean(maze_results["false_cors"]))
        #print(pearsonr([p[1] for p in maze_results["perfs"]], maze_results["true_cors"]))
        
    data = np.array([all_true, all_false]).T # (2, num_mazes, num_pairs)

    plt.figure(figsize = (1.6,2))

    xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
    jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
    plt.bar(xs, ms, yerr = ss, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2}, color = cols)
    for idata, datapoints in enumerate(data.T):
        plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
        
    plt.xticks(xs, ["same\nmaze", "different\nmaze"])#, rotation = 45, ha ="right")
    plt.ylabel("correlation of W with A")
    plt.axhline(0.0, color = "k")
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.ylim(0.0, 1.0)
    plt.yticks([0, 1])
    plt.savefig(f"{basefigdir}all_adjacency_matrix_correlations{iref}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()



#%% Plot within-maze and across-maze slot similarity

cols = [np.array(plt.get_cmap("tab10")(0)) for _ in range(2)]
cols[1] = np.zeros(3)+0.6

sames, diffs = [], []

for datadir in datadirs:

    sub_data = pickle.load(open(f"{datadir}sub_data_split.pickle", "rb")) # save the trial data
    Csubs = sub_data["Csubs"] # the Csub matrices

    for i1 in range(len(Csubs)):
        for i2 in range(len(Csubs)):
            sim = (Csubs[i1][0] * Csubs[i2][1]).sum(-1)[:-1].mean(-1)
            if i1 == i2:
                sames.append(sim)
            else:
                diffs.append(sim)
                

sames, diffs = np.array(sames), np.array(diffs)

true_cors, false_cors = sames.mean(-1), diffs.mean(-1)

mtrue, strue = np.mean(true_cors), np.std(true_cors)
bins_false = np.linspace(np.amin(false_cors)-1e-5, np.amax(false_cors)+1e-5, 12)
bins_true = np.linspace(np.amin(true_cors)-1e-5, np.amax(true_cors)+1e-5, 6)

plt.figure(figsize = (2.5,1.5))
countst, _, _ = plt.hist(true_cors, bins = bins_true, label = "same maze", density = True, color = cols[0])
countsf, _, _ = plt.hist(false_cors, bins = bins_false, label = "different maze", density = True, color = cols[1])

ymin, ymax = 0, np.ceil(max(np.amax(countsf), np.amax(countst))/5)*5
plt.ylim(ymin, ymax)
plt.xlabel(r"subspace correlation")
plt.ylabel("frequency")
plt.yticks([])
plt.legend(frameon = False)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}slot_overlaps{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



#%%
model_results = [pickle.load(open(f"{datadir}decoder_generalization_performance.pickle", "rb")) for datadir in datadirs]
    
all_res = np.array([pysta.utils.compute_model_support(result) for result in model_results])
model_scores = all_res[..., 1] - all_res[..., 0] # performance when satisfing cond minus performance when not
#model_scores = [np.mean(res[1]) - np.mean(res[0]) for res in all_res] # performance when satisfing cond minus performance when not
mean, std = np.mean(model_scores, axis = 0), np.std(model_scores, axis = 0)
                    
xs = np.arange(len(mean))
plt.figure(figsize = (1.6,2.0))
plt.bar(xs, mean, yerr = std)
plt.axhline(0.0, color = "k")
plt.gca().spines[['right', 'top']].set_visible(False)
plt.yticks([-0.1, 0.1, 0.3, 0.5, 0.7])
plt.ylabel("pattern overlap")
plt.xticks(xs, ["relative", "absolute"], rotation = 45, ha ="right") 
plt.ylim(-0.1, 0.7)
plt.savefig(f"{basefigdir}decoding_model_support{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()
    


#%% look at future position coding

labels = ["states", "transitions"]
for idata, data_type in enumerate(["", "transition_"]):

    all_data_true = [pickle.load(open(f"{datadir}decoder_{data_type}generalization_performance.pickle", "rb")) for datadir in datadirs]
    all_data_ref = [pickle.load(open(f"{datadir}decoder_{data_type}generalization_performance.pickle", "rb")) for datadir in datadirs_ref]

    plan_perfs, plan_xs = [], []
    ex_perfs, ex_xs = [], []
    test_neural = -1
    for all_data in [all_data_true, all_data_ref]:

        neural_ts, loc_ts = [list(np.array([data[key] for data in all_data]).mean(0).astype(int)) for key in ["neural_times", "loc_times"]]

        perfs = np.array([data["nongen_scores"] for data in all_data])


        plan_perfs.append(perfs[:, neural_ts.index(test_neural), 1:])
        plan_xs.append(loc_ts[1:])

        deltas = np.arange(-4, 5)
        delta_perfs = [[] for _ in range(len(deltas))]
        for idelta, delta in enumerate(deltas):
            for i1, t1 in enumerate(neural_ts):
                for i2, t2 in enumerate(loc_ts):
                    if t1 >= 0 and (t2 - t1 == delta):
                        delta_perfs[idelta].append(perfs[:, i1, i2])

        ex_perfs.append(np.array([np.mean(np.array(perf), 0) for perf in delta_perfs]).T)
        ex_xs.append(deltas)


    plt.figure(figsize = (2, 1.5))
    for i in range(2):
        xs, ms, ss = plan_xs[i], np.mean(plan_perfs[i], 0), np.std(plan_perfs[i], 0)
        plt.plot(xs, ms)
        plt.fill_between(xs, ms-ss, ms+ss, alpha = 0.2)
    plt.xlabel("time in future")
    plt.ylabel("accuracy", labelpad = -5)
    plt.yticks([0, 1])
    plt.ylim(0, 1)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{basefigdir}future_{labels[idata]}_planning{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

    plt.figure(figsize = (2.3, 1.5))
    for i in range(2):
        xs, ms, ss = ex_xs[i], np.mean(ex_perfs[i], 0), np.std(ex_perfs[i], 0)
        ms[xs == 0], ss[xs == 0] = np.nan, np.nan
        plt.plot(xs, ms)
        plt.fill_between(xs, ms-ss, ms+ss, alpha = 0.2)
    plt.fill_between([-1, +1], [0, 0], [1,1], color = "k", alpha = 0.07, zorder = -10, linewidth = 0)
    plt.xlabel("relative time")
    plt.ylabel("accuracy", labelpad = -5)
    plt.yticks([0, 1])
    plt.ylim(0, 1)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{basefigdir}future_{labels[idata]}_execution{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


# %% now plot wall input to locations or non-locations in the subspace

analysis_results = [pickle.load(open(f"{datadir}wall_to_transition_result.pickle","rb")) for datadir in datadirs]     
# analysis_results = [pickle.load(open(f"{datadir}wall_to_transition_result.pickle","rb")) for datadir in datadirs_ref]             


cols = {"stim": (0.35, 0.65, 0.2),
          "strong_ex": plt.get_cmap("coolwarm")(0.95), 
          "weak_ex": plt.get_cmap("coolwarm")(0.55),
          "strong_inh": plt.get_cmap("coolwarm")(0.05),
          "weak_inh": plt.get_cmap("coolwarm")(0.35),
          "neutral": plt.get_cmap("coolwarm")(0.44),}



#%% first plot effective connectivity


endpoints, adjacents, others = [np.array([result[key] for result in analysis_results]) for key in ["connection_to_endpoint", "connection_to_adjacent", "connection_to_other"]]

data = np.array([endpoints, adjacents, others]).mean(-1).T
plotcols = [cols[key] for key in ["strong_ex", "weak_inh"]] + [np.zeros(3)+0.6]
xticks = ["consistent", "adjacent", "other"]

plt.figure(figsize = (2.75,2.1))

xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
plt.bar(xs, ms, yerr = ss, color = plotcols, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
    
plt.xticks(xs, xticks)#, rotation = 45, ha = "right")
plt.ylabel("effective connectivity")
plt.yticks([0,1,2])
plt.axhline(0.0, color = "k", lw = 1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}effective_transition_connectivity{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% then wall input to different transitions within subspace

walls, adjacents, others = [np.array([result[key] for result in analysis_results]).mean(-1) for key in ["input_to_wall", "input_to_adjacent", "input_to_other"]]

plotcols2 = plotcols
plotcols2[0] = cols["strong_inh"]
data = np.array([walls, adjacents, others]).T

plt.figure(figsize = (2.75,2.1))

xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
plt.bar(xs, ms, yerr = ss, color = plotcols2, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
    
plt.ylabel("wall input projection")
#plt.xticks([0, 1], ["wall-adjacent", "reference"], rotation = 45, ha = "right")
plt.xticks(xs, ["wall", "adjacent", "other"])
plt.axhline(0.0, color = "k", lw = 1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}wall_to_transition_input{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


# %%
