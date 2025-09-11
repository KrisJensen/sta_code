

#%%
# import numpy as np
import torch
import matplotlib.pyplot as plt
import pysta
import pickle
import numpy as np
from pysta import basedir
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
pysta.reload()

ext = ".pdf"

#%%

model_name = "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model22"
datadir = f"{basedir}/data/rnn_analyses/" + "_".join(model_name.split("/")) + "_"
datadir_sta = f"{basedir}/data/rnn_analyses/sta_" + "_".join(model_name.split("/")) + "_"

result = pickle.load(open(f"{datadir}fixed_point_analyses.p", "rb"))
sta_result = pickle.load(open(f"{datadir_sta}fixed_point_analyses.p", "rb"))
ctrl_result = pickle.load(open(f"{datadir}fixed_point_analyses_ctrl.p", "rb"))

#%%
proj_all_acts, proj_old_r, proj_new_rs, deltas, relax_deltas, walls, strengths, all_all_acts, old_r, deltas_raw, num_ps, paths = [result[k] for k in ["proj_all_acts", "proj_old_r", "proj_new_rs", "deltas", "relax_deltas", "walls", "strengths", "all_all_acts", "old_r", "deltas_raw", "num_ps", "paths"]]
sta_deltas, sta_relax_deltas, sta_strengths, sta_all_acts, sta_old_r, sta_strengths, sta_new_rs = [sta_result[k] for k in ["deltas", "relax_deltas", "strengths", "all_all_acts", "old_r", "strengths", "proj_new_rs"]]

ctrl_deltas, ctrl_deltas_raw = [ctrl_result[k] for k in ["deltas", "deltas_raw"]]


#%% plot example representation before stimulation, after weak stimulation, and after strong stimulation

ind = 0
to_plot = [proj_old_r, proj_new_rs[1], proj_new_rs[-1]]
#to_plot = [sta_old_r, sta_new_rs[1], sta_new_rs[-1]]
for ir, r in enumerate(to_plot):
    ax = pysta.plot_utils.plot_perspective_attractor(walls, r[ind][:, :], plot_proj = True, cmap = "YlOrRd", figsize = (3.5,2.2), aspect = (1,1,4.2), view_init = (-30,-10,-90),
    filename = f"{basedir}/figures/attractor/example_stim{ir}.pdf",  show = True, bbox_inches = mpl.transforms.Bbox([[0.75,0.7], [2.8,1.54]]))


#%% plot the change in representation vs stim magnitude

plot_deltas = deltas[:, ind]
half_ind = np.where(plot_deltas > 0.5*plot_deltas[-1])[0][0]
plot_deltas = plot_deltas[:2*half_ind]
plot_deltas_ctrl = ctrl_deltas[:, ind][:2*half_ind]

sta_plot_deltas = sta_deltas[:, ind]
sta_half_ind = np.where(sta_plot_deltas > 0.5*sta_plot_deltas[-1])[0][0]
sta_plot_deltas = sta_plot_deltas[:2*sta_half_ind]

xs = np.linspace(0, 1, len(plot_deltas))
sta_xs = np.linspace(0, 1, len(sta_plot_deltas))

rnn_col = np.array(plt.get_cmap("tab10")(0))
sta_col = np.minimum(1.0, rnn_col*1.5)
sta_col = np.array([49, 204, 199])/255
cols = [sta_col, rnn_col, np.ones(3)*0.6]

plt.figure(figsize = (1.5, 1.5))
plt.plot(sta_xs, sta_plot_deltas, label = "STA", color = cols[0], zorder = 1)
plt.plot(xs, plot_deltas, label = "RNN", color = cols[1], zorder = 2)
plt.plot(xs, plot_deltas_ctrl, color = cols[2], label = "ctrl", zorder = 0)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("stim strength")
plt.ylabel("representational\nchange")
plt.xticks([])
plt.xlim(0, 1)
plt.legend(frameon = False, handletextpad = 0.5, handlelength = 0.6, loc = "lower center", bbox_to_anchor = (0.85, 0.1))
plt.savefig(f"{basedir}/figures/attractor/change_by_stim.pdf", bbox_inches = "tight", transparent = True)
plt.show()

#%% also raw
plot_deltas_raw = deltas_raw[:, ind]
plot_deltas_raw_ctrl = ctrl_deltas_raw[:, ind]

xs = np.linspace(0, 1, len(plot_deltas_raw))

plt.figure(figsize = (1.5, 1.5))
plt.plot(xs, plot_deltas_raw, color = cols[1], label = "RNN")
plt.plot(xs, plot_deltas_raw_ctrl, color = cols[2], label = "ctrl")
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("stim strength")
plt.ylabel("rate change")
plt.xticks([])
plt.xlim(0, 1)
plt.legend(frameon = False, handletextpad = 0.5, handlelength = 0.6, loc = "lower center", bbox_to_anchor = (0.85, 0.1))
plt.savefig(f"{basedir}/figures/attractor/change_by_stim_raw.pdf", bbox_inches = "tight", transparent = True)
plt.show()

#%% now plot example neural trajectories

strength_inds = np.array([0, 5, 8, 12, len(diffs)-1])

diffs = np.abs(proj_all_acts - proj_old_r[None, None, :proj_all_acts.shape[2], ...]).sum((-1, -2))

T = proj_all_acts.shape[1]
ts = np.linspace(0, 1, T)

plt.figure(figsize = (2,1.5))
for strength_ind in strength_inds[::-1]:
    col = plt.get_cmap("viridis")(strength_ind / (1+np.amax(strength_inds)))
    plt.plot(ts, diffs[strength_ind, :, ind], color = col)

ymin, ymax = -0.2, np.amax(diffs[strength_inds, ...])*1.08
plt.fill_between(np.array([num_ps[0], num_ps[0]+num_ps[1]])/np.sum(num_ps), [ymin, ymin], [ymax, ymax], zorder = -10, color = "k", alpha = 0.10, lw = 0)
plt.gca().spines[['right', 'top', 'bottom']].set_visible(False)
plt.xlabel("time")
plt.ylim(ymin, ymax)
plt.xticks([])
plt.xlim(0, 1)
plt.ylabel("representational\nchange")
plt.savefig(f"{basedir}/figures/attractor/change_by_time.pdf", bbox_inches = "tight", transparent = True)
plt.show()


#%% now plot example raw neural trajectories


#strength_inds = [0, 3,7, 8, len(strengths)-2]
diffs_raw = ((all_all_acts - old_r.numpy()[None, None, :all_all_acts.shape[2], ...])**2).sum((-1, -2))

plt.figure(figsize = (2,1.5))
for strength_ind in strength_inds[::-1]:
    col = plt.get_cmap("viridis")(strength_ind / (1+np.amax(strength_inds)))
    plt.plot(ts, diffs_raw[strength_ind, :, ind], color = col)

ymin, ymax = -2, np.amax(diffs_raw[strength_inds, ...])*1.08
plt.fill_between(np.array([num_ps[0], num_ps[0]+num_ps[1]])/np.sum(num_ps), [ymin, ymin], [ymax, ymax], zorder = -10, color = "k", alpha = 0.10, lw = 0)
plt.gca().spines[['right', 'top', 'bottom']].set_visible(False)
plt.xlabel("time")
plt.ylim(ymin, ymax)
plt.xticks([])
plt.xlim(0, 1)
plt.ylabel("rate change")
plt.savefig(f"{basedir}/figures/attractor/raw_by_time.pdf", bbox_inches = "tight", transparent = True)
plt.show()


#%% and for the STA
sta_strength_inds = np.array([0, 3,6,10])
sta_diffs_raw = ((sta_all_acts - sta_old_r.numpy()[None, None, :sta_all_acts.shape[2], ...])**2).sum((-1, -2))
#sta_diffs_raw = ((np.log(sta_all_acts) - np.log(sta_old_r.numpy()[None, None, ...]))**2).sum((-1, -2))
sta_ts = np.linspace(0, 1, sta_diffs_raw.shape[1])

plt.figure(figsize = (2,1.5))
for strength_ind in sta_strength_inds[::-1]:
    col = plt.get_cmap("viridis")(strength_ind / (1+np.amax(sta_strength_inds)))
    plt.plot(sta_ts, sta_diffs_raw[strength_ind, :, ind], color = col)

ymin, ymax = -0.2, np.amax(sta_diffs_raw[sta_strength_inds, ...])*1.08
plt.fill_between(np.array([num_ps[0], num_ps[0]+num_ps[1]])/np.sum(num_ps), [ymin, ymin], [ymax, ymax], zorder = -10, color = "k", alpha = 0.10, lw = 0)
plt.gca().spines[['right', 'top', 'bottom']].set_visible(False)
plt.xlabel("time")
plt.ylim(ymin, ymax)
plt.xticks([])
plt.xlim(0, 1)
plt.ylabel("representational\nchange")
plt.savefig(f"{basedir}/figures/attractor/sta_by_time.pdf", bbox_inches = "tight", transparent = True)
plt.show()

#%% plot a scaffold with the two paths

adjacency = pysta.maze_utils.compute_adjacency(walls)[0].numpy()
num_locs = adjacency.shape[0]
L = int(np.sqrt(num_locs))


plt.figure(figsize = (2.4, 2.4))
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adjacency, ax = ax)
for ipath, path in enumerate(paths):
    locs = np.array(pysta.maze_utils.index_to_loc(np.array(path), L)).astype(float)
    smooth_locs = gaussian_filter1d(np.repeat(locs, 6, axis = -1), 2, axis = -1, mode = "nearest").T
    if ipath == 0:
        ax.plot(smooth_locs[:, 0], smooth_locs[:, 1], color = plt.get_cmap("viridis")(0.9), lw = 7, ls = "-", zorder = 10, label = "better")
    else:
        ax.plot(smooth_locs[:, 0], smooth_locs[:, 1], color = plt.get_cmap("viridis")(0.7), lw = 7, ls = "-", label = "worse")#dashes=(3, 2.5))
#ax.plot(smooth_locs[:, 0], smooth_locs[:,1])
ax.axis("off")
plt.legend(frameon = False, loc = "upper center", bbox_to_anchor = (0.5, 1.18), ncol = 2)
plt.savefig(f"{basedir}/figures/attractor/paths{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% plot a colorbar

plt.figure(figsize = (0.6,4))
plt.imshow(np.linspace(+1, -1, 101)[:, None], vmin = -1, vmax = +1, aspect = "auto", cmap = "viridis")
plt.xticks([])
plt.yticks([])
plt.savefig(f"{basedir}/figures/attractor/stim_cbar{ext}", bbox_inches = "tight", transparent = True)
plt.show()


# %%
