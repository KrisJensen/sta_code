

#%%

import pysta
from pysta import basedir
import pickle
import matplotlib.pyplot as plt
import numpy as np
ext = ".pdf"

seeds = [21,24,25]
basetasks = ["static_relrew", "static_planrew", "moving_relrew", "moving_planrew"]
labels = ["RNN", "STA"]
basenames = {"RNN": {"static_relrew": "MazeEnv_L4_max6_goal_changing-rew_static-rew_constant-maze_allo_relrew_plan5-6-7_VanillaRNN_iter9-10-11_tau5.0_opt_N800_linout_model",
                "static_planrew": "MazeEnv_L4_max6_goal_changing-rew_static-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
                "moving_relrew": "MazeEnv_L4_max6_goal_changing-rew_dynamic-rew_constant-maze_allo_relrew_plan5-6-7_VanillaRNN_iter9-10-11_tau5.0_opt_N800_linout_model",
                "moving_planrew": "MazeEnv_L4_max6_goal_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",}
            }

basenames["STA"] = {key: "sta_"+value for (key, value) in basenames["RNN"].items()}
min_dist, max_dist = 3, 6

sequence_colors = [plt.get_cmap("viridis")(iind / 5 + 0.35) for iind in range(4)][::-1]
figsize = (2.5, 1.5)

#%%

for basetask in basetasks:
    results = []
    for label in labels:
        datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(basenames[label][basetask].split("/")) + str(seed) + "_" for seed in seeds]
        #results.append(np.array([pickle.load(open(f"{datadir}simple_{basetask}_decode_from_planning_minmax{min_dist}-{max_dist}.pickle", "rb"))[0]["nongen_scores"] for datadir in datadirs]))
        results.append(np.array([pickle.load(open(f"{datadir}simple_{basetask}_decoder_generalization_performance_minmax{min_dist}-{max_dist}.pickle", "rb"))["nongen_scores"] for datadir in datadirs]))
    results = np.array(results)

    xs = np.arange(results.shape[-1])[1:]
    ineural = 0 # from end of planning
    plt.figure(figsize = (2.5,2))
    for imodel in range(results.shape[0]):
        data = results[imodel, :, ineural, 1:]
        ms, ss = data.mean(0), data.std(0)
        plt.plot(xs, ms, label = labels[imodel])
        plt.fill_between(xs, ms-ss, ms+ss, alpha = 0.2)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend(frameon = False)
    plt.ylim(0,1)
    plt.xticks([1,2,3])
    print(f"{basetask} t={ineural-1}")
    plt.xlabel("time in trial")
    plt.ylabel("prediction accuracy")
    plt.savefig(f"{pysta.basedir}/figures/simple_tasks/state_at_time_decoding_{basetask}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
        
#%%

for basetask in basetasks:
    results = []
    for label in labels:
        datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(basenames[label][basetask].split("/")) + str(seed) + "_" for seed in seeds]
        results.append(np.array([pickle.load(open(f"{datadir}simple_{basetask}_decode_from_planning_minmax{min_dist}-{max_dist}.pickle", "rb")) for datadir in datadirs]))
    results = np.array(results)
    
    results = np.nanmean(results, axis = (-1,-2))
    
    ms, ss = results.mean(-1), results.std(-1)
    xs = np.arange(len(ms))
    plt.figure(figsize = (1.8,2))
    plt.bar(xs, ms, yerr = ss, color = [plt.get_cmap("tab10")(x) for x in xs])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.ylim(0,1)
    plt.xticks(xs, labels)
    print(f"{basetask}")
    plt.ylabel("prediction accuracy")
    plt.savefig(f"{pysta.basedir}/figures/simple_tasks/state_any_time_decoding_{basetask}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


