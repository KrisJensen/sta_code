
import torch
import numpy as np

def get_biases(Nrec, num_locs):
    b_fwd, b_bwd = [torch.zeros(Nrec, 1) for _ in range(2)]
    b_fwd[:num_locs, 0] = 1/num_locs
    b_bwd[-num_locs:, 0] = 1/num_locs
    return b_fwd, b_bwd

def get_fp(G_fwd, G_bwd, log_omega, tau = 10, max_iters = 500, thresh = 1e-4, Print = False, p = None, clamp = None):
    """find the fixed point q for a given set of weights"""
    batch, num_modules, num_locs = log_omega.shape
    Nrec = num_modules * num_locs
    b_fwd, b_bwd = get_biases(Nrec, num_locs)
    
    flat_log_omega = log_omega.reshape(batch, -1, 1)
    clip_minval = torch.tensor(1e-40)
    iter_ = -1
    z = torch.log(torch.ones(batch, Nrec, 1) / num_locs)
    q = z.exp()
    eps = torch.inf
    while eps > thresh and (iter_ < max_iters):
        iter_ += 1
        dzdt = flat_log_omega + torch.maximum(clip_minval, G_fwd @ q + b_fwd).log() + torch.maximum(clip_minval, G_bwd @ q + b_bwd).log()

        z = (1-1/tau) * z + dzdt / tau
        
        #normalise by slot
        z = z.reshape(batch, num_modules, num_locs)
        z = z - z.logsumexp(axis = -1, keepdims = True) # normalise
        z = z.reshape(batch, Nrec, 1)

        qold = q
        q = torch.clip(z.exp(), 0.0, 100.0)
        if clamp is not None:
            q[:, clamp, :] = p[:, clamp, :]

        eps = (q-qold).abs().max()
        if Print and (iter_ % 10 == 0):
            print(iter_, eps)
            
    if iter_ == max_iters:
        if np.random.uniform() < 0.025:
            print("max iters reached, eps =", eps)

    return q

def fwd_bwd(T, omega, action_max = False):
    """T is the transition matrix, omega is the likelihood"""
        
    batch, num_modules, num_locs = omega.shape

    alphas = torch.zeros(batch, num_modules, num_locs)
    alphas[:, 0, :] = omega[:, 0, :].clone()
    betas = torch.ones(batch, num_modules, num_locs) / num_locs # beta_T = uniform

    for t in range(1, num_modules):
        if action_max:
            A = (T > 1e-2).to(float) # reachable states
            # compute most likely path that passes through x at time t
            alphas[:, t, :] = omega[:, t, :] * torch.amax(A * alphas[:, t-1, :][:, None, :], axis = -1) # max over x_t-1 and a_t-1
            betas[:, num_modules-t-1, :] = torch.amax(A.transpose(-1,-2) * (betas[:, num_modules-t, :] * omega[:, num_modules-t, :])[:, None, :], axis = -1)

        else:
            alphas[:, t, :] = omega[:, t, :] * (T @ alphas[:, t-1, :, None])[..., 0] # sum over t-1
            betas[:, num_modules-t-1, :] = (T.transpose(-1,-2) @ (betas[:, num_modules-t, :, None] * omega[:, num_modules-t, :, None]))[..., 0]

        # normalise
        alphas[:, t, :] /= alphas[:, t, :].sum(-1, keepdims = True)
        betas[:, num_modules-t-1, :] /= betas[:, num_modules-t-1].sum(-1, keepdims = True)

    # compute marginal posterior probability of passing through x at time t
    posterior = alphas*betas
    posterior /= posterior.sum(axis = -1, keepdims = True)

    return posterior.reshape(batch, -1, 1)

def get_loglik(obs, num_locs, beta = 5):
    batch = obs.shape[0]
    obs_mod = obs.reshape(batch, -1, num_locs) # module by module
    rew_func = obs_mod[:, 1:-2, :].clone() # reward function (batch, modules, locs)
    rew_func[:, 0, :] += 10.0*obs_mod[:, 0, :] # very strong signal to start at current loc
    log_rew_func = rew_func * beta
    log_rew_func = log_rew_func - torch.logsumexp(log_rew_func, axis = -1, keepdims = True)
    
    return log_rew_func

def KL(p, q, num_locs = 16):
    """p and q are (batch, Nrec, 1)"""
    return torch.mean(p * (torch.log(p) - torch.log(q))) * num_locs

def eval(env, T, G_fwd, G_bwd, beta = 5, reps = 5, jitter = 1e-10, action_max = False, fixed_omega = None, clamp = None, p = None):
    num_locs = env.num_locs
    losses = []
    for _ in range(reps):
        if fixed_omega is None:
            env.reset()
            log_omega = get_loglik(env.observation(), num_locs, beta = beta)
        else:
            log_omega = fixed_omega
        
        p = fwd_bwd(T, log_omega.exp(), action_max = action_max)
        q = get_fp(G_fwd, G_bwd, log_omega, p = p, clamp = clamp)

        losses.append(KL(p + jitter, q + jitter))
    return np.mean(losses)

def block_norm(M, num_locs):
    """normalise each block of rows"""
    num_modules = int(M.shape[0] / num_locs)
    Mnorm = M.clone()
    for t in range(num_modules):
        inds = torch.arange(num_locs) + t*num_locs
        Mnorm[inds, :] *=  num_locs / (1e-20 + Mnorm[inds, :].sum())
    return Mnorm

def get_mask_mats(Nrec, num_locs, num_modules):
    bwd_mask_mat = torch.zeros(Nrec, Nrec)
    for i in range(num_modules):
        bwd_mask_mat[num_locs*i:num_locs*(i+1), num_locs*(i+1):] = 1
    fwd_mask_mat = bwd_mask_mat.T
    return fwd_mask_mat, bwd_mask_mat

def mask(M, mask_mat, num_locs, Gthresh = torch.tensor(1e-40)):
    M = torch.maximum(Gthresh, M)
    M = mask_mat*M
    M = block_norm(M, num_locs)
    return M

def ADAM_update(G, m, v, dG, t, lrate, beta1 = 0.9, beta2 = 0.99, eps = 1e-8):
    m = beta1 * m + (1.0 - beta1) * dG
    v = beta2 * v + (1.0 - beta2) * (dG**2)
    m_hat = m / (1.0 - beta1 ** (t+1))
    v_hat = v / (1.0 - beta2 ** (t+1))
    G += lrate * m_hat / (torch.sqrt(v_hat) + eps)
    return G, m, v