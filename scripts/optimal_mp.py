"""
thoughts:
1) option of normalised (update eqs) or unnormalised (include diagonal elements)
2) verify bias has no effect in STA
3) consider row-normalisation of the Gs and verify it doesn't change anything
4) optionally use 'maximising' targets (i.e. max over states/actions)

"""



#%%
# 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pysta
import torch
import copy
from scipy.stats import pearsonr


seed = 31
model_name = f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model{seed}"


rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = True, create_dp = False) # load the model
    
env = rnn.env
num_modules = rnn.num_modules
num_locs = env.num_locs
batch = env.batch
A = env.adjacency
T = A / torch.sum(A, axis = 1, keepdims = True)
Nrec = rnn.Nrec
beta = 10

np.savetxt("adjacency.csv", A[0].to(int), delimiter = ",")


b_fwd, b_bwd = [torch.zeros(Nrec, 1) for _ in range(2)]
b_fwd[:num_locs, 0] = 1/num_locs
b_bwd[-num_locs:, 0] = 1/num_locs

#%%

def get_lik(obs, beta = 10, log = False):
    obs_mod = obs.reshape(env.batch, -1, env.num_locs) # module by module
    rew_func = obs_mod[:, 1:-2, :].clone() # reward function (batch, modules, locs)
    rew_func[:, 0, :] += 10.0*obs_mod[:, 0, :] # very strong signal to start at current loc
    log_rew_func = rew_func * beta
    log_rew_func = log_rew_func - torch.logsumexp(log_rew_func, axis = -1, keepdims = True)
    
    if log:
        return log_rew_func
    else:
        return log_rew_func.exp()
     
def fwd_bwd(T, omega, normalise = True):
        
    batch, num_modules, _ = omega.shape

    alphas = torch.zeros(batch, num_modules, env.num_locs)
    alphas[:, 0, :] = omega[:, 0, :].clone()

    betas = torch.ones(env.batch, num_modules, env.num_locs) / env.num_locs # beta_T = uniform

    for t in range(1, num_modules):

        # compute most likely path that passes through x at time t
        alphas[:, t, :] = omega[:, t, :] * (T @ alphas[:, t-1, :, None])[..., 0] # sum over t-1
        betas[:, num_modules-t-1, :] = (T.transpose(-1,-2) @ (betas[:, num_modules-t, :, None] * omega[:, num_modules-t, :, None]))[..., 0]

        # normalize
        if normalise:
            alphas[:, t, :] /= alphas[:, t, :].sum(-1, keepdims = True)
            betas[:, num_modules-t-1, :] /= betas[:, num_modules-t-1].sum(-1, keepdims = True)

    posterior = alphas*betas
    
    if normalise:
        posterior /= posterior.sum(axis = -1, keepdims = True)

    return posterior

def plot_Gs(G_fwd, G_bwd):
    plot_G = G_fwd+G_bwd
    #plot_G[(fwd_mask(torch.ones(Nrec, Nrec)) + bwd_mask(torch.ones(Nrec, Nrec))) < 1e-2] = torch.nan
    pysta.plot_utils.plot_slot_connectivity(plot_G, num_locs, show = True, figsize = (5,5), vmin = 0.0, vmax = 1.0, transparent = True, yticks = [str(i) for i in range(0, num_modules)], xticks = [str(i) for i in range(0, num_modules)], xtickrot = 0)

def eval(env, G_fwd, G_bwd, normalise, beta, reps = 5, jitter = 1e-10):
    losses = []
    for _ in range(reps):
        log_omega = get_lik(env.observation(), beta=beta, log = True)
        q = get_fp(G_fwd, G_bwd, log_omega, normalise = normalise)
        p = fwd_bwd(T, log_omega.exp(), normalise = normalise).reshape(batch, Nrec, 1)
        KL = KL_norm if normalise else KL_unorm
        losses.append(KL(p + jitter, q + jitter))
    return np.mean(losses)

def block_norm(M):
    Mnorm = M.clone()
    for t in range(num_modules):
        inds = torch.arange(num_locs) + t*num_locs
        Mnorm[inds, :] *=  num_locs / (1e-20 + Mnorm[inds, :].sum())
    return Mnorm


#%%
env.reset()
obs = env.observation()
omega = get_lik(obs, beta = 10)
posterior = fwd_bwd(T, omega)

flat_post = posterior.reshape(env.batch, -1)

actions = posterior[:, 1, :].argmax(-1)

opt_acts = env.optimal_actions().argmax(-1)
print((actions == opt_acts).to(float).mean())

# %%

def get_STA_weights(T):
    Gsta_fwd, Gsta_bwd = [torch.zeros(batch, Nrec, Nrec) for _ in range(2)]
    # set recurrent weights
    for mod1 in range(num_modules-1): # for each subspace
        # indices of the weight matrices corresponding to this subspace and the next
        mod1_inds, mod2_inds = [torch.arange(num_locs)+num_locs*ind for ind in [mod1, mod1+1]]
        for i1, ind1 in enumerate(mod1_inds):
            for i2, ind2 in enumerate(mod2_inds):
                # weights are just the adjacency matrix plus some noise
                Gsta_fwd[:, ind2, ind1] = T.sqrt()[:, i2, i1].clone()
                Gsta_bwd[:, ind1, ind2] = T.transpose(-1, -2).sqrt()[:, i1, i2].clone()
    
    
    return Gsta_fwd, Gsta_bwd

def get_fp(G_fwd, G_bwd, log_omega, tau = 20, max_iters = 200, thresh = 1e-5, Print = False, normalise = True):

    flat_log_omega = log_omega.reshape(batch, -1, 1)
    clip_minval = torch.tensor(1e-30)
    iter_ = -1
    z = torch.log(torch.ones(batch, Nrec, 1) / num_locs)
    q = z.exp()
    eps = torch.inf
    while eps > thresh and (iter_ < max_iters):
        iter_ += 1
        dzdt = flat_log_omega + torch.maximum(clip_minval, G_fwd @ q + b_fwd).log() + torch.maximum(clip_minval, G_bwd @ q + b_bwd).log()

        z = (1-1/tau) * z + dzdt / tau
        
        if normalise:
            #normalise by slot
            z = z.reshape(batch, num_modules, num_locs)
            z = z - z.logsumexp(axis = -1, keepdims = True) # normalise
            z = z.reshape(batch, Nrec, 1)

        qold = q
        q = torch.clip(z.exp(), 0.0, 100.0)

        eps = (q-qold).abs().max()
        if Print and (iter_ % 10 == 0):
            print(iter_, eps)

    return q

def KL_norm(p, q):
    return torch.sum(p * (torch.log(p) - torch.log(q))) / (batch*num_modules)

def KL_unorm(p, q):
    return KL_norm(p, q) + (q.sum() - p.sum()) / (batch*num_modules)


Gsta_fwd, Gsta_bwd = get_STA_weights(T)
plot_Gs(Gsta_fwd[0], Gsta_bwd[0])

torch.manual_seed(1)
np.random.seed(1)

env.reset()
log_omega = get_lik(env.observation(), beta=5, log = True)
q = get_fp(Gsta_fwd, Gsta_bwd, log_omega)
p = fwd_bwd(T, log_omega.exp()).reshape(batch, Nrec, 1)

for rep in [p, q]:
    vmap = rep.reshape(batch, num_modules, num_locs)
    plot_kwargs = {"walls": env.walls[0], "vmap": vmap[0], "loc": env.loc[0], "goal": None, "aspect": (1,1,6)}
    pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = 0.0, vmax = 1.0, figsize = (8,5))
    
    actions = posterior[:, 1, :].argmax(-1)

    print((vmap[:, 1, :].argmax(-1) == env.optimal_actions().argmax(-1)).to(float).mean())

print(eval(env, Gsta_fwd, Gsta_bwd, True, beta))



#%%

normalise = True
bwd_mask_mat = torch.zeros(Nrec, Nrec)

if normalise:
    KL = KL_norm
    for i in range(num_modules):
        bwd_mask_mat[16*i:16*(i+1), 16*(i+1):] = 1
else:
    KL = KL_unorm
    for i in range(num_modules):
        bwd_mask_mat[16*i:16*(i+1), 16*i:] = 1
        #bwd_mask_mat[16*i:16*(i+1), 16*(i+1):] = 1
fwd_mask_mat = bwd_mask_mat.T

np.random.seed(0)
torch.manual_seed(0)

def mask(M, mode = "fwd", Gthresh = torch.tensor(1e-25)):
    
    M = torch.maximum(Gthresh, M)
    if mode == "fwd":
        M = fwd_mask_mat*M
    else:
        M = bwd_mask_mat*M
    if normalise:
        M = block_norm(M)

    return M

fwd_mask = lambda M: mask(M, mode = "fwd")
bwd_mask = lambda M: mask(M, mode = "bwd")


G_fwd, G_bwd = [f(torch.ones(Nrec, Nrec)/num_locs) for f in [fwd_mask, bwd_mask]]
m_fwd, v_fwd, m_bwd, v_bwd = [torch.zeros_like(G_fwd) for _ in range(4)]

plot_Gs(G_fwd, G_bwd)
beta = 5
lrate = 1e-4

for iter_ in range(50000):
    
    env.reset()
    log_omega = get_lik(env.observation(), beta, log = True)
    omega = log_omega.exp()
    p = fwd_bwd(T, omega, normalise = normalise).reshape(batch, -1, 1)
    
    q = get_fp(G_fwd, G_bwd, log_omega, normalise = normalise)
    q_thresh = q + 1e-20

    if iter_ % 100 == 0:
        print(iter_, eval(env, G_fwd, G_bwd, normalise, beta, reps = 20))

    diag_omega = torch.diag_embed(omega.reshape(batch, -1))
    u_fwd = torch.diag_embed((G_fwd @ q + b_fwd)[..., 0])
    u_bwd = torch.diag_embed((G_bwd @ q + b_bwd)[..., 0])

    if normalise:
        r = torch.diagonal(diag_omega * u_fwd * u_bwd, dim1 = -2, dim2 = -1)

        P = torch.zeros(batch, Nrec, Nrec)
        for t in range(num_modules):
            inds = torch.arange(num_locs) + t*num_locs
            Zt = r[:, inds].sum(-1)[:, None, None]
            P[:, inds[:, None], inds[None, :]] = (torch.eye(num_locs) - q[:, inds, :] @ torch.ones(1, num_locs) )/Zt
        pq = (p / q_thresh )
    else:
        pq = (p / q_thresh ) - 1
        P = torch.eye(Nrec)
    
    J = torch.eye(Nrec) - P @ diag_omega @ (u_bwd @ G_fwd + u_fwd @ G_bwd)
    a = torch.linalg.solve(torch.transpose(J, -1, -2), pq)
    PTa =  torch.transpose(P, -1, -2) @ a
    
    qT = torch.transpose(q, -1, -2)

    dG_fwd = ((diag_omega @ (u_bwd @ PTa)) @ qT).mean(0)
    dG_bwd = ((diag_omega @ (u_fwd @ PTa)) @ qT).mean(0)
    
    if torch.isnan(torch.mean(dG_fwd+dG_bwd)):
        print('nan update')
        break
    
    # ADAM
    beta1, beta2, eps = 0.9, 0.99, 1e-8
    m_fwd = beta1 * m_fwd + (1.0 - beta1) * dG_fwd
    v_fwd = beta2 * v_fwd + (1.0 - beta2) * (dG_fwd**2)
    m_fwd_hat = m_fwd / (1.0 - beta1 ** (iter_+1))
    v_fwd_hat = v_fwd / (1.0 - beta2 ** (iter_+1))
    G_fwd += lrate * m_fwd_hat / (torch.sqrt(v_fwd_hat) + eps)
    
    m_bwd = beta1 * m_bwd + (1.0 - beta1) * dG_bwd
    v_bwd = beta2 * v_bwd + (1.0 - beta2) * (dG_bwd**2)
    m_bwd_hat = m_bwd / (1.0 - beta1 ** (iter_+1))
    v_bwd_hat = v_bwd / (1.0 - beta2 ** (iter_+1))
    G_bwd += lrate * m_bwd_hat / (torch.sqrt(v_bwd_hat) + eps)
    
    # constrain to positive upper/lower triangular!!!
    G_fwd = fwd_mask(G_fwd)
    G_bwd = bwd_mask(G_bwd)
    
    # since we're taking a product, we can scale one up and the other down arbitrarily.
    # we just keep the scales consistent
    rescale = (G_fwd.mean() / G_bwd.mean()).sqrt()
    G_fwd = G_fwd / rescale
    G_bwd = G_bwd * rescale


pickle.dump([G_fwd, G_bwd], open("Gs.p", "wb"))

#%% visualise an example path

plot_Gs(G_fwd, G_bwd)

for rep in [p, q]:
    vmap = rep.reshape(batch, num_modules, num_locs)
    vmap = vmap / vmap.sum(-1, keepdims = True) # normalise to plot
    plot_kwargs = {"walls": env.walls[0], "vmap": vmap[0], "loc": env.loc[0], "goal": None, "aspect": (1,1,6)}
    pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = vmap.min(), vmax = vmap.max(), figsize = (8,5))
    
    actions = posterior[:, 1, :].argmax(-1)

    print((vmap[:, 1, :].argmax(-1) == env.optimal_actions().argmax(-1)).to(float).mean())


# %% now plot similarity of different connectivity matrices to different powers of the adjacency matrix

Csubs = np.zeros((num_modules, num_locs, Nrec))
baserange = np.arange(rnn.env.num_locs)
for i in range(len(Csubs)):
    inds = baserange + i * rnn.env.num_locs
    Csubs[i, baserange, inds] = 1

abs_deltas = np.arange(0, 4) # subspace differences to look at
deltas = np.arange(-abs_deltas.max(), abs_deltas.max() + 1) # both positive and negative deltas
all_Wavgs, all_scales, all_scores, all_Arefs, all_scores_abs, all_Wavgs_abs = [], [], [], [], [], []


Wrec_m = (G_fwd+G_bwd).detach().numpy()
Wrec_m[np.isnan(Wrec_m)] = 0.0
adj_m = T[0].sqrt()
# compute powers of the adjacency matrix for comparison. Open question whether to only use the sign. Should investigate empirically what the RNN learns.
Arefs = [torch.matrix_power(torch.tensor(adj_m), i).numpy() for i in range(4)]
Wavgs, Wavgs_abs = [], []

abs_scores = np.zeros((len(abs_deltas), len(Arefs))) # effective weight matrix match to each adjacency matrix power
scales = np.zeros((len(abs_deltas))) # also look at the 'scale' (std) of the connectivity for each delta
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
    Weff = np.nanmean(Weffs, axis = 1) # average across pairs of subspaces
    Weff_abs = np.nanmean(Weff, 0) # also average positive and negative
    
    Wavgs.append(Weff)
    Wavgs_abs.append(Weff_abs)

    # compute the point-biserial correlation
    abs_scores[idelta, :] = np.array([pearsonr(Weff_abs.flatten(), A.flatten())[0] for A in Arefs])
    for isign in range(2):
        scores[isign, idelta, :] = np.array([pearsonr(Weff[isign, ...].flatten(), [A, A.T][isign].flatten())[0] for A in Arefs])
    
    # compute std
    scales[idelta] = Weffs.std((-1,-2)).mean() # average across pairs of subspaces and sign
        
    all_scales.append(scales)
    all_scores.append(scores)
    all_scores_abs.append(abs_scores)
    all_Wavgs.append(Wavgs)
    all_Wavgs_abs.append(Wavgs_abs)
    all_Arefs.append(Arefs)


#%% plot the average weight matrix for different distances

vmin, vmax = np.nanquantile(Wavgs_abs, [0.10, 0.95])
for imat, mat in enumerate(Wavgs_abs):
    plt.figure(figsize = (3,3))
    plt.imshow(mat, cmap = "coolwarm", vmin = vmin, vmax = vmax)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{abs_deltas[imat]}")
    plt.show()
    plt.close()


vmin, vmax = np.nanquantile(Wavgs, [0.10, 0.95])
for imat, mat in enumerate(Wavgs):
    for isign in range(2):
        plt.figure(figsize = (3,3))
        plt.imshow(mat[isign], cmap = "coolwarm", vmin = vmin, vmax = vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{abs_deltas[imat]}-{isign}")
        plt.show()
        plt.close()


#%% now plot the similarity plots
figsize = (2.5, 1.5)
for isign in range(3):
    plot_deltas = abs_deltas
    if isign == 0:
        plot_scores = np.array(all_scores_abs)
    elif isign == 1:
        plot_scores = np.array(all_scores)[:, 0, ...]
    else:
        plot_scores = np.array(all_scores)[:, 1, ...]

    # plot the similarity of each effective weight matrix to each reference power of the adjacency matrix
    plt.figure(figsize = figsize)
    for idelta, score in enumerate(np.transpose(plot_scores, (1,0,2))):
        if idelta > 0:
            m, s = score.mean(axis = 0), score.std(axis = 0)
            plt.plot(plot_deltas, m, label = f"{idelta}")
            plt.fill_between(plot_deltas, m-s, m+s, alpha = 0.2, linewidth = 0)
    plt.xlabel("order of transition matrix", labelpad = 3.5)
    plt.ylabel("similarity", labelpad = 2.5)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 4,
            handlelength = 1.2, handletextpad = 0.5, columnspacing = 0.8, frameon = False)
    plt.xticks(plot_deltas)
    plt.xlim(0,3)
    plt.show()
    plt.close()
    
    print(plot_scores)
# %%
# plot the parameter scale
plt.figure(figsize = figsize)
m, s = np.array(all_scales).mean(0), np.array(all_scales).std(0)
plt.plot(abs_deltas, m, color = "k")
plt.fill_between(abs_deltas, m-s, m+s, alpha = 0.2, color = "k", linewidth = 0)
plt.xlabel("subspace difference")
plt.ylabel("average strength")
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xticks(abs_deltas)
plt.show()
plt.close()

# %%
