#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pysta
import torch
from scipy.ndimage import gaussian_filter1d
import copy
from scipy.stats import pearsonr
ext = ".pdf"
basefigdir = f"{pysta.basedir}/figures/optimal_mp/"
np.random.seed(0)
torch.manual_seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{pysta.basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8
figsize = (1.85, 1.6)

#%% load all data

beta = 5.5

# seeds = [31,32,33,34,35]
# results = [pickle.load(open(f"{pysta.basedir}/data/optimal_mp/beta7.0_result_seed{seed}.p", "rb")) for seed in seeds]

# seeds = [31,32,33,34,35]
# results = [pickle.load(open(f"{pysta.basedir}/data/optimal_mp/beta5.0_iter15000_result_seed{seed}.p", "rb")) for seed in seeds]

seeds = [31,32,33,34,35]
results = [pickle.load(open(f"{pysta.basedir}/data/optimal_mp/beta{beta}_result_seed{seed}.p", "rb")) for seed in seeds]


# seeds = [31,32,33,34,35]
# results = [pickle.load(open(f"{pysta.basedir}/data/optimal_mp/beta6.0_result_seed{seed}.p", "rb")) for seed in seeds]


# seeds = [31]
# results = [pickle.load(open(f"{pysta.basedir}/data/optimal_mp/fixed_beta7.0_iter2000_result_seed{seed}.p", "rb")) for seed in seeds]


#% plot full connectivity matrix
ex = 0
G_fwd, G_bwd, T = results[ex]["G_fwd"], results[ex]["G_bwd"], results[ex]["T"]
num_locs, num_modules, Nrec = [results[ex][key] for key in ["num_locs", "num_modules", "Nrec"]]

masks = torch.sum(torch.stack(pysta.mp_utils.get_mask_mats(Nrec, num_locs, num_modules)), axis = 0)
masks[masks == 0] = np.nan
print(results[0]["beta"])
pysta.plot_utils.plot_slot_connectivity(G_fwd+G_bwd+masks, num_locs, show = True, figsize = (3.5,3.5), vmin = 0.0, vmax = 1.0, transparent = True, yticks = [str(i) for i in range(0, num_modules)], xticks = [str(i) for i in range(0, num_modules)], xtickrot = 0, filename = f"{basefigdir}/example_weights{ext}", xlabel = None, ylabel = None)



#%% plot learning curves

all_KLs = np.array([result["KLs"] for result in results])
smooth_KLs = gaussian_filter1d(all_KLs, 10, axis = 1)

m, s = np.mean(smooth_KLs, axis = 0), np.std(smooth_KLs, axis = 0)/np.sqrt(smooth_KLs.shape[0])
xs = np.arange(len(m))*50

plt.figure(figsize = (1.65, 1.1))
plt.plot(xs, m)
plt.fill_between(xs, m-s, m+s, alpha = 0.2, linewidth = 0)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("epoch", labelpad = 3.5)
plt.ylabel("KL[p||q]", labelpad = 2.5)
plt.xlim(-xs[-1]*0.01, xs[-1])
plt.xticks([0, 5000, 10000, 15000])
plt.savefig(f"{basefigdir}training_curves{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



#%% visualise an example path

model_name = results[ex]["name"]
rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = True, create_dp = False)
env = rnn.env
env.reset()
beta = results[ex]["beta"]

log_omega = pysta.mp_utils.get_loglik(env.observation(), num_locs, beta = beta)
p = pysta.mp_utils.fwd_bwd(T, log_omega.exp())
q = pysta.mp_utils.get_fp(G_fwd, G_bwd, log_omega)

bbox_inches = mpl.transforms.Bbox([[2.0,1.35], [5.1,2.65]])

for imap, vmap in enumerate([p, q]):
    trial = 3
    vmap = vmap.reshape(vmap.shape[0], num_modules, num_locs)
    plot_kwargs = {"walls": env.walls[trial], "vmap": vmap[trial], "loc": env.loc[trial], "goal": None, "aspect": (1,1,6)}
    pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = vmap.min(), vmax = vmap.max(), figsize = (7,4), filename = f"{basefigdir}/example_representation_{['p', 'q'][imap]}{ext}", bbox_inches = bbox_inches)
    
    print((vmap[:, 1, :].argmax(-1) == env.optimal_actions().argmax(-1)).to(float).mean())


# %% now plot similarity of different connectivity matrices to different powers of the adjacency matrix

Csubs = np.zeros((num_modules, num_locs, Nrec))
baserange = np.arange(rnn.env.num_locs)
for i in range(len(Csubs)):
    inds = baserange + i * rnn.env.num_locs
    Csubs[i, baserange, inds] = 1

max_delta = 3
abs_deltas = np.arange(max_delta+1) # only look at absolute value of delta for now, since positive and negative are similar
deltas = np.arange(max_delta, max_delta + 1) # both positive and negative deltas
all_Wavgs, all_scales, all_scores, all_Arefs = [], [], [], []

for iseed, seed in enumerate(seeds):
    result = results[iseed] # pickle.load(open(f"{pysta.basedir}/data/optimal_mp/result_seed{seed}.p", "rb"))
    G_fwd, G_bwd, T = result["G_fwd"], result["G_bwd"], result["T"]
    Wrec_m = (G_fwd+G_bwd).detach().numpy()
    Wrec_m[np.isnan(Wrec_m)] = 0.0
    adj_m = T[0]#.sqrt()
    #adj_m = (T[0] > 0.1).to(float)

    Arefs = [torch.matrix_power(torch.tensor(adj_m), i).numpy() for i in range(4)]
    Wavgs, Wavgs_abs = [], []

    scores = np.zeros((2, len(abs_deltas), len(Arefs)))
    for idelta, abs_delta in enumerate(abs_deltas): # for each subspace difference
        Weffs = [] # initialize effective weight matrix
        for isign, delta in enumerate([-abs_delta, +abs_delta]): # consider positive and negative together
            Weffs.append([])
            for mod0 in range(num_modules): # go through subspaces
                mod1 = mod0 + delta # for each paired subspace
                if min(mod0, mod1) >= 0 and max(mod0, mod1) < num_modules: # only consider pairs that are in range
                    Weffs[-1].append(Csubs[mod1] @ Wrec_m @ Csubs[mod0].T)
                
        Weffs = np.array(Weffs)

        # normalise
        #Weffs = (Weffs - Weffs.mean((-1,-2), keepdims = True)) / Weffs.std((-1,-2), keepdims = True) # normalise weight matrix by its mean and std
        
        Weff = np.nanmean(Weffs, axis = 1) # average across pairs of subspaces

        Wavgs.append(Weff)

        # compute the point-biserial correlation
        for isign in range(2):
            scores[isign, idelta, :] = np.array([pearsonr(Weff[isign, ...].flatten(), [A, A.T][isign].flatten())[0] for A in Arefs])
        
        scales = np.zeros((2, 3, 3))
        for i1, ind1 in enumerate(range(4, 7)):
            for idelta, delta in enumerate(range(1,4)):
                ind2, ind3, ind4 = (ind1 - delta), (7-ind1), (7-ind1 + delta)
                
                inds1 = np.arange(num_locs) + ind1 * num_locs
                inds2 = np.arange(num_locs) + ind2 * num_locs
                scales[0, i1, idelta] = Wrec_m[inds1, :][:, inds2].sum()
                
                inds3 = np.arange(num_locs) + ind3 * num_locs
                inds4 = np.arange(num_locs) + ind4 * num_locs
                scales[1, i1, idelta] = Wrec_m[inds3, :][:, inds4].sum()
                #print(ind1, ind2, scales[0, i1, idelta], ind3, ind4, scales[1, i1, idelta])
        
        
    all_scales.append(scales)
    all_scores.append(scores)
    all_Wavgs.append(Wavgs)
    all_Arefs.append(Arefs)


#%% plot the average weight matrix for different distances
plot_Ws = np.concatenate([np.array(all_Arefs[ex])[:,None,...], np.array(all_Wavgs[ex])], 1)
vmin, vmax = np.nanquantile(Wavgs, [0.00, 1.00])
for imat, mat in enumerate(plot_Ws):
    fig, axs = plt.subplots(1,3,figsize=(4.3,1.15))
    for isign in range(3):
        ax = axs[[0,2,1][isign]]
        v_min, v_max = (0, np.amax(mat[isign])) if isign == 0 else (vmin, vmax)
        v_min, v_max = np.amin(mat[isign]), np.amax(mat[isign])
        ax.imshow(mat[isign], cmap = "coolwarm", vmin = v_min, vmax = v_max)
        ax.set_xticks([])
        ax.set_yticks([])
        #if imat == 1:
        #    ax.set_title(['structure', 'backward', 'forward'][isign])
    plt.savefig(f"{basefigdir}/Wavg_{abs_deltas[imat]}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


#%% now plot the similarity plots
figsize = (2.5, 1.5)
for isign in range(2):
    if isign == 0:
        plot_scores = np.array(all_scores)[:, 0, ...]
    else:
        plot_scores = np.array(all_scores)[:, 1, ...]

    # plot the similarity of each effective weight matrix to each reference power of the adjacency matrix
    plt.figure(figsize = figsize)
    for idelta, score in enumerate(np.transpose(plot_scores, (1,0,2))):
        if idelta > 0:
            m, s = score.mean(axis = 0), score.std(axis = 0)/np.sqrt(score.shape[0])
            plt.plot(abs_deltas, m, label = f"{idelta}")
            plt.fill_between(abs_deltas, m-s, m+s, alpha = 0.2, linewidth = 0)
    plt.xlabel("power of transition matrix", labelpad = 3.5)
    plt.gca().spines[['right', 'top']].set_visible(False)
    if isign == 1:
        plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 0.3), ncol = 4, handlelength = 1.7, handletextpad = 0.7, columnspacing = 1.3, frameon = False)
        plt.ylabel("similarity", labelpad = 2.5)
    else:
        plt.yticks([])
    plt.xticks(abs_deltas)
    plt.xlim(0,3)
    plt.ylim(0,1)
    plt.savefig(f"{basefigdir}/similarity_{['bwd', 'fwd'][isign]}{ext}", bbox_inches = "tight", transparent = True)

    plt.show()
    plt.close()
    
    print(plot_scores.mean(0))
    
# %%
# plot the parameter scale
all_scales = np.array(all_scales)
norm_scales = all_scales / np.sum(all_scales, axis = -1, keepdims = True) # sum across offsets
mean_scales = norm_scales.mean(-2) # mean across rows


plt.figure(figsize = (2.1, 1.5))
for i in range(2):
    m, s = np.array(mean_scales[:, i, :]).mean(0), np.array(mean_scales[:, i, :]).std(0) / np.sqrt(mean_scales.shape[0])
    plt.plot(range(1,4), m, label = ["forward", "backward"][i])
    plt.fill_between(range(1,4), m-s, m+s, alpha = 0.2, linewidth = 0)
plt.xlabel("subspace difference")
plt.ylabel("normalised\nconnection strength")
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xticks(range(1,4))
plt.xlim(1,3)
plt.ylim(0,1)
plt.legend(frameon = False)
plt.savefig(f"{basefigdir}/scale{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

# %%

# %%


result_clamp = pickle.load(open(f"{pysta.basedir}/data/optimal_mp/clamp_beta7.0_iter20000_result_seed31.p", "rb"))
plt.figure(figsize = (2.5, 1.8))
plt.plot(np.arange(401)*200*50, gaussian_filter1d(result_clamp["KLs"], sigma=5))
plt.xlabel("epoch")
plt.ylabel(r'$KL[p_{t=3}||q_{t=3}]$')
plt.show()

G = (result_clamp["G_fwd"] + result_clamp["G_bwd"])[3*16:4*16, :]
pysta.plot_utils.plot_slot_connectivity(G, num_locs, show = True, figsize = (4.5,1), vmin = 0.0, vmax = 1.0, transparent = True, yticks = [], xticks = [str(i) for i in range(0, num_modules)], xtickrot = 0, xlabel = "input", ylabel = "output")

print([np.round(G[:, torch.arange(16) + i*16].sum().item()/16, 3) for i in range(7)])


# %%
