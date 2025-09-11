
#%%

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from pysta import basedir
import copy
pysta.reload()
ext = ".pdf"

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%%

data = pickle.load(open(f"{basedir}/data/examples/example_task_data.pickle", "rb"))
cmap = "viridis"
vmin, vmax = -2.3, 1.15
no_loc = True

# plot moving goal
moving_goal = data["moving_goal"]
moving_goal2 = copy.deepcopy(moving_goal)
moving_goal2["optimal_actions"], moving_goal2["loc"] = moving_goal["optimal_actions_t2"], moving_goal["loc_t2"]
    
pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/moving_goal1{ext}", vmap = moving_goal["rew"][0], **moving_goal, cmap = cmap, vmin = vmin, vmax = vmax, xlabel = None, show = True)
pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/moving_goal2{ext}", vmap = moving_goal2["rew"][1], goal_step_num = 1, **moving_goal2, cmap = cmap, vmin = vmin, vmax = vmax, xlabel = None, show = True)

# plot static goal
static_goal = data["static_goal"]
pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/static_goal{ext}", vmap = static_goal["rew"][0], **static_goal, cmap = cmap, vmin = vmin, vmax = vmax, show = True)

# plot reward landscape at different times
rew_land = data["rew_landscape"]
for i in range(4):
    rew_land_i = copy.deepcopy(rew_land)
    rew_land_i["loc"] = rew_land["all_locs"][i]
    rew_land_i["optimal_actions"] = None if i >= 2 else rew_land["optimal_actions"]
    pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/rew_land{i+1}{ext}", vmap = rew_land["rews"][i], **rew_land_i, cmap = cmap, vmin = vmin, vmax = vmax, xlabel = None, show = True)

#%%
plt.figure(figsize = (0.6,4))
plt.imshow(np.linspace(+1, -1, 101)[:, None], vmin = vmin, vmax = vmax, aspect = "auto", cmap = cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(f"{basedir}/figures/handcrafted_examples/rew_cbar{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% now plot example representations

data = pickle.load(open(f"{basedir}/data/examples/example_rep_data.pickle", "rb"))
env_labels = ["static", "changing", "moving", "landscape"]
env_titles = ["constant goal", "changing goal", "moving goal", "reward landscape"]

cmap_vals = {"td": [0.2, 1.0], "sr":[-1.5, 1.], "sta": [0, 0.8]}
for agent in ["td", "sr", "sta"]:
    for env in env_labels:
        vmin, vmax = cmap_vals[agent]
        kwargs = data[env][agent]
        fname = f"{basedir}/figures/handcrafted_examples/{env}_{agent}{ext}"

        # renormalize slightly
        kwargs["vmap"] = np.maximum(np.minimum(kwargs["vmap"], vmax), vmin)
        kwargs["vmap"] = (kwargs["vmap"] - vmin) / (vmax - vmin)
        
        if agent == "sta":
            kwargs["vmap"] = kwargs["vmap"][:-1, :]
            pysta.plot_utils.plot_perspective_attractor(filename = fname, **kwargs, cmap = "YlOrRd", vmin = -0.15, vmax = 1.2, show = True, goal_inds = range((0 if env == "moving" else 1), kwargs["vmap"].shape[0]), bbox_inches = mpl.transforms.Bbox([[2,1.45], [5.3,2.54]]))
        else:
            if env == "changing" and agent == "td":
                vmin, vmax = 0.15, 0.25
            elif env == "moving" and agent == "td":
                vmin, vmax = 0.20, 0.30
            else:
                vmin, vmax = -0.15, 1.2
            pysta.plot_utils.plot_flat_frame(filename = fname, **kwargs, cmap = "YlOrRd", vmin = vmin, vmax = vmax, show = True)



# %%
