
#%%
# 
import torch
import numpy as np
import pickle
import pysta
from pysta.mp_utils import get_fp, fwd_bwd, get_loglik, eval, mask, ADAM_update, get_biases, get_mask_mats
import time
import sys
pysta.reload()


def run_optimal_mp(seed, beta = 6, action_max = False, max_iters = 20000, fixed_omega = False, clamp = False):

    # load our model of interest
    
    basestr = f"action_beta{beta}_" if action_max else f"beta{beta}_"
    basestr = ("fixed_" if fixed_omega else "") + basestr
    basestr = ("clamp_" if clamp else "") + basestr

    print(f"\n\n\nseed={seed}")
    print(basestr)
    # load model
    model_name = f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model{seed}"
    rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = True, create_dp = False)

    # extract and set some parameters
    env = rnn.env
    num_modules, num_locs, batch, Nrec = rnn.num_modules, rnn.num_locs, rnn.batch, rnn.Nrec
    
    # optionally clamp some things
    clamp_inds = torch.cat([torch.arange(num_locs)+num_locs*i for i in [0,1,2,4,5,6]]) if clamp else None

    # extract adjacency and initialise 'biases'
    A = env.adjacency
    assert len(A.shape) == 3 # batch, locs, locs
    T = A / torch.sum(A, axis = 1, keepdims = True) # T_ij = p(x_t = i|x_t-1 = j)
    b_fwd, b_bwd = get_biases(Nrec, num_locs) # construct bias vectors
    
    # create masks for upper/lower block triangular parameters
    fwd_mask_mat, bwd_mask_mat = get_mask_mats(Nrec, num_locs, num_modules)
    
    # store some useful information and results
    result = {"name": model_name, "batch": batch, "num_modules": num_modules, "num_locs": num_locs, "Nrec": Nrec, "A": A, "T": T, "beta": beta}
    

    #%% run optimisation

    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialise parameters and ADAM moments
    G_fwd, G_bwd = [mask(torch.ones(Nrec, Nrec)/num_locs, mask_mat, num_locs) for mask_mat in [fwd_mask_mat, bwd_mask_mat]]
    m_fwd, v_fwd, m_bwd, v_bwd = [torch.zeros_like(G_fwd) for _ in range(4)]

    lrate = 1e-4
    t0 = time.time()
    KLs = []
    for iter_ in range(max_iters):
        
        # get likelihood and true posterior
        if fixed_omega:
            if iter_ == 0:
                env.reset()
                log_omega = get_loglik(env.observation(), num_locs, beta = beta)
                log_omega[...] = log_omega[:1, ...] # fixed likelihood across batches
        else:
            env.reset()
            log_omega = get_loglik(env.observation(), num_locs, beta = beta)
            
        omega = log_omega.exp()
        p = fwd_bwd(T, omega, action_max = action_max)
        
        # get approximate posterior
        q = get_fp(G_fwd, G_bwd, log_omega, p = p, clamp = clamp_inds)
        q_thresh = q + 1e-40

        # occasionally print progress
        if iter_ % 50 == 0:
            eval_KL = eval(env, T, G_fwd, G_bwd, beta = beta, reps = 10, action_max = action_max, fixed_omega = log_omega, clamp = clamp_inds)
            KLs.append(eval_KL)
            print(iter_, eval_KL, ", t =", np.round((time.time() - t0)/60, 2), "mins")
            sys.stdout.flush()
            if iter_ % 1000 == 0:
                result["KLs"], result["G_fwd"], result["G_bwd"]  = KLs, G_fwd, G_bwd
                pickle.dump(result, open(f"{pysta.basedir}/data/optimal_mp/{basestr}iter{iter_}_result_seed{seed}.p", "wb"))
            
        # diagonal embeddings of likelihood and fwd/bwd messages
        diag_omega = torch.diag_embed(omega.reshape(batch, -1))
        u_fwd = torch.diag_embed((G_fwd @ q + b_fwd)[..., 0])
        u_bwd = torch.diag_embed((G_bwd @ q + b_bwd)[..., 0])

        # normalisation Jacobian
        r = torch.diagonal(diag_omega * u_fwd * u_bwd, dim1 = -2, dim2 = -1)
        P = torch.zeros(batch, Nrec, Nrec)
        for t in range(num_modules):
            inds = torch.arange(num_locs) + t*num_locs
            Zt = r[:, inds].sum(-1)[:, None, None]
            P[:, inds[:, None], inds[None, :]] = (torch.eye(num_locs) - q[:, inds, :] @ torch.ones(1, num_locs) )/Zt
        
        # compute fixed points gradients
        pq = (p / q_thresh )
        J = torch.eye(Nrec) - P @ diag_omega @ (u_bwd @ G_fwd + u_fwd @ G_bwd)
        a = torch.linalg.solve(torch.transpose(J, -1, -2), pq)
        PTa =  torch.transpose(P, -1, -2) @ a
        
        qT = torch.transpose(q, -1, -2)
        dG_fwd = ((diag_omega @ (u_bwd @ PTa)) @ qT).mean(0)
        dG_bwd = ((diag_omega @ (u_fwd @ PTa)) @ qT).mean(0)
        
        # check that everything is finite
        if torch.isnan(torch.mean(dG_fwd+dG_bwd)):
            print('nan update')
            result["KLs"], result["G_fwd"], result["G_bwd"]  = KLs, G_fwd, G_bwd
            pickle.dump(result, open(f"{pysta.basedir}/data/optimal_mp/{basestr}result_break_seed{seed}.p", "wb"))
            break
        
        # ADAM update
        G_fwd, m_fwd, v_fwd = ADAM_update(G_fwd, m_fwd, v_fwd, dG_fwd, iter_, lrate)
        G_bwd, m_bwd, v_bwd = ADAM_update(G_bwd, m_bwd, v_bwd, dG_bwd, iter_, lrate)
        
        # constrain to positive upper/lower triangular
        G_fwd = mask(G_fwd, fwd_mask_mat, num_locs)
        G_bwd = mask(G_bwd, bwd_mask_mat, num_locs)

    # store final parameters
    result["KLs"], result["G_fwd"], result["G_bwd"]  = KLs, G_fwd, G_bwd
    pickle.dump(result, open(f"{pysta.basedir}/data/optimal_mp/{basestr}result_seed{seed}.p", "wb"))

if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        beta = float(sys.argv[1].split("-")[-1].split("_")[0])
        action_max = bool(int(sys.argv[2].split("-")[-1]))
        seed = int(sys.argv[3].split("-")[-1].split("_")[0])

        print(beta, action_max, seed)
        run_optimal_mp(seed, beta = beta, action_max = action_max)
    else:
        run_optimal_mp(31, beta = 6.1, action_max = False)

