
#%%
import pysta
import numpy as np
from sklearn.decomposition import PCA
import pickle
from pysta import basedir

#%%

basenames = ["MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model",
             "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_relrew_plan5-6-7/VanillaRNN/iter9-10-11_tau5.0_opt/N800_linout/model",
             "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model"]

baselabels = ["WM", "relrew", "sta"]
seeds = [31,32,33,34,35]
results = []

for ibase, basename in enumerate(basenames):
    seed_results = []
    for seed in seeds:

        rnn, figdir, datadir = pysta.utils.load_model(basename+str(seed), create_sta = (baselabels[ibase] == "sta"), create_dp = False) # load the model
        trial_data = pickle.load(open(f"{datadir}trial_data.pickle", "rb")) # load the trial data

        #data_name = basename.replace("/", "_") + str(seed)
        #connectivity_data_p = pickle.load(open(f"{basedir}/data/rnn_analyses/{data_name}_planning_subspaces.pickle", "rb"))
        #connectivity_data_e = pickle.load(open(f"{basedir}/data/rnn_analyses/{data_name}_connectivity_data.pickle", "rb"))
        connectivity_data_p = pickle.load(open(f"{datadir}planning_subspaces.pickle", "rb"))
        connectivity_data_e = pickle.load(open(f"{datadir}connectivity_data.pickle", "rb"))


        Csubs_p = np.concatenate([C for C in connectivity_data_p["Csubs"]], axis = 0)
        Csubs_e = np.concatenate([C for C in connectivity_data_e["Csubs"]], axis = 0)

        all_vars = [[], [], []]

        for t in range(trial_data["rs"].shape[1]):
            r = trial_data["rs"][:, t, :]
            r = r - r.mean(0) # center the data

            pca = PCA().fit(r)

            proj_p = r @ Csubs_p.T # projecton onto each component
            proj_e = r @ Csubs_e.T # projecton onto each component

            base_var = r.var(0).sum() # total variance in the data
            var_exp_p = proj_p.var(0)
            var_exp_e = proj_e.var(0)
            var_pca = pca.explained_variance_ratio_
            
            # plt.figure()
            # plt.plot(var_pca)
            # plt.plot(var_exp_p / base_var, ls = "--")
            # plt.plot(var_exp_e / base_var, ls = "--")
            # plt.axhline(1/r.shape[-1], color = np.zeros(3)+0.7)
            # plt.xlim(0, 100)
            # plt.ylim(0, var_pca[0])
            # plt.show()

            print(ibase, seed, t, var_pca[:var_exp_p.size].sum(), var_exp_p.sum()/base_var, var_exp_e.sum()/base_var)

            for iv, var in enumerate([var_pca, var_exp_p/base_var, var_exp_e/base_var]):
                all_vars[iv].append(var)
        seed_results.append(all_vars)
    
    seed_results = [np.array([seed[i] for seed in seed_results]) for i in range(3)]
    results.append(seed_results)

pickle.dump([results, trial_data["step_nums"][0, :, 0]], open(f"{basedir}/data/rnn_analyses/subspace_variance_explained.pickle", "wb"))


# %%

# ncomp = 100
# var_exp = np.zeros(ncomp)
# for i in range(ncomp):
#     rp = r - proj[:, i, None] * C[None, i, :]
#     var_exp[i] = base_var - rp.var(0).sum()
#     if i % 25 == 0:
#         print(i, var_exp[i])
