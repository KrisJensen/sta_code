#%%
import pysta
from pysta import basedir
import pickle
import numpy as np
import torch

results = {}

np.random.seed(0)
torch.manual_seed(0)

def update_goal(env, goal, scale_rew = 1.0):
    env.goal = torch.zeros(env.batch, env.max_steps+1, dtype = int) + goal
    env.rews = torch.zeros(env.batch, env.max_steps+1, env.num_locs) + env.rew_nogoal # default negative reward
    for t in range(env.max_steps+1):
        env.rews[env.batch_inds, t, env.goal[:, t]] = env.rew_goal # positive reward at the goal location
    env.rews *= scale_rew
    env.compute_value_function()
    
def update_env(env, walls, goal, loc, scale_rew = 1.0):
    env.sample_maze(walls = walls)
    env.loc = env.loc*0+loc
    update_goal(env, goal, scale_rew)
    
def walls_from_barriers(barriers):
    
    walls = torch.zeros(env.batch, env.num_locs, 4)
    for barrier in barriers:
        reverse = (env.nowall_neighbors[barrier[0], barrier[1]], [1,0,3,2][barrier[1]])
        for wall in [barrier, reverse]:
            walls[:, wall[0], wall[1]] = 1
    return walls

def set_initial_path(sta, initial_path, initial_path_inds):
    for ip, p in enumerate(initial_path):
        sta.z0[ip, p] += 100.0 * (ip in initial_path_inds)
    sta.z0 -= sta.z0.logsumexp(dim = -1, keepdim = True)
    sta.reset()

def compare_paths(sta, env, expected_paths, niters, initial_path = [], initial_path_inds = []):
    
    fits = np.zeros((niters, 2))
    examples = []

    for iter_ in range(niters):
        if iter_ % 10 == 0:
            print(iter_, np.mean(fits[:iter_], axis = 0))
        
        set_initial_path(sta, initial_path, initial_path_inds)
        update_env(env, walls, goal, loc, scale_rew = 1.0)
        action = sta.step(env.observation())

        activity = sta.r[0].detach().numpy()
        min_fits = [np.amin([activity[ip, p] for (ip, p) in enumerate(path)]) for path in expected_paths]
        for ifit, fit in enumerate(min_fits):
            if fit > 0.9:
                fits[iter_, ifit] = 1
                if fits[:, ifit].sum() == 1:
                    examples.append(sta.r[0].detach().numpy())
                
    print(np.mean(fits, axis = 0))
    return fits, examples

aspect, figsize = (1,1,5), (8,5)
iters_per_action = 1201
# initialize an environment with a stationary goal (change 'dynamic_rew' to 'True' for a moving goal)
print("\nInstantiating environment")
env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_rew = False, batch_size = 1, dynamic_rew = False, max_steps = 4, side_length = 3)


#%% first we run simple analyses on an environment with no walls

loc, goal = 0, 8
walls = torch.zeros(env.batch, env.num_locs, 4)

update_env(env, walls, goal, loc, scale_rew = 1.0)
sta = pysta.agents.SpaceTimeAttractor(env,  beta = 9.0, tau = 50, iters_per_action = iters_per_action, shift_time = 2.0, rec_noise = 1e-1, adj_noise = 0.0, adj_bias = 0.0)
sta.store_all_activity = True
update_env(env, walls, goal, loc, scale_rew = 1.0)
env.plot(show = True)
observation = env.observation()
action = sta.step(observation)

results["empty"] = {"rs": [sta.phi(sta.z0)[None, ...]]+sta.all_acts[0], "walls": walls}
#%% plot result
for ind in [0, 50, 200, 500]:
    r = sta.all_acts[0][ind][0]
    plot_kwargs = {"walls": env.walls[0], "vmap": r, "loc": env.loc[0], "goal": env.goal[0], "aspect": aspect}
    pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = figsize)


#%% now we run an analysis on an environment with two paths
loc, goal = 0, 8
barriers = [(1,2),(3,2),(4,0)]
walls = walls_from_barriers(barriers)
#expected_paths = [[0,1,2,5,8], [0,3,6,7,8]]
expected_paths = [[0,1,4,5,8], [0,3,6,7,8]]

update_env(env, walls, goal, loc, scale_rew = 1.0)
sta = pysta.agents.SpaceTimeAttractor(env,  beta = 9.0, tau = 50, iters_per_action = iters_per_action, shift_time = 2.0, rec_noise = 1e-0, adj_noise = 0.0, adj_bias = 0.0)
update_env(env, walls, goal, loc, scale_rew = 1.0)
env.plot(show = True)

fits, examples = compare_paths(sta, env, expected_paths, 1000)

#%% also generate full trajectory
sta.store_all_activity = True
sta.reset()
update_env(env, walls, goal, loc, scale_rew = 1.0)
action = sta.step(env.observation())

for ind in [0, 140, 700, 1000]:
    r = sta.all_acts[0][ind][0]
    plot_kwargs = {"walls": env.walls[0], "vmap": r, "loc": env.loc[0], "goal": env.goal[0], "aspect": aspect}
    pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = figsize)


results["twopaths"] = {"example_convergence": [sta.phi(sta.z0)[None, ...]]+sta.all_acts[0], "fixed_points": examples,
                       "fits": fits, "walls": walls}



#%% and remove reward input

update_env(env, walls, goal, loc, scale_rew = 0.0)
action = sta.step(env.observation())

plot_kwargs = {"walls": env.walls[0], "vmap": sta.r[0].detach().numpy(), "loc": env.loc[0], "goal": env.goal[0], "aspect": aspect}
pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = figsize)


results["twopaths"]["norew"] = sta.r[0].detach().numpy()



#%% now we look at an example with two minima of different quality

env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_rew = False, batch_size = 1, dynamic_rew = False, max_steps = 5, side_length = 3)

loc, barriers, goal = 0, [(1,0),(3,2),(4,0)], 5
walls = walls_from_barriers(barriers)
expected_paths = [[0,1,2,5], [0,3,6,7,8,5]]
update_env(env, walls, goal, loc, scale_rew = 1.0)
sta = pysta.agents.SpaceTimeAttractor(env,  beta = 9.0, tau = 50, iters_per_action = iters_per_action, shift_time = 2.0, rec_noise = 1e-0, adj_noise = 0.0, adj_bias = 0.0)
update_env(env, walls, goal, loc, scale_rew = 1.0)
env.plot(show = True)

initial_path = expected_paths[1]
initial_path_inds = [2]
fits1, examples1 = compare_paths(sta, env, expected_paths, 1000, initial_path = [], initial_path_inds = [])
fits2, examples2 = compare_paths(sta, env, expected_paths, 1000, initial_path = initial_path, initial_path_inds = initial_path_inds)


sta = pysta.agents.SpaceTimeAttractor(env,  beta = 9.0, tau = 50, iters_per_action = iters_per_action, shift_time = 2.0, rec_noise = 1e-0, adj_noise = 0.0, adj_bias = 0.0)
rnormal = sta.r[0].detach().numpy()
set_initial_path(sta, initial_path, initial_path_inds)
rdiff = sta.r[0].detach().numpy()
update_env(env, walls, goal, loc, scale_rew = 1.0)

for r in [examples1[0], examples2[0], rnormal, rdiff]:
    plot_kwargs = {"walls": env.walls[0], "vmap": r, "loc": env.loc[0], "goal": env.goal[0], "aspect": aspect}
    pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = figsize)



results["goodbad"] = {"fixed_points": [examples1[0], examples2[0], rnormal, rdiff],
                      "fits": [fits1, fits2], "walls": walls}


#%%

pickle.dump(results, open(f"{basedir}/data/examples/fixed_points.p", "wb"))


# %%
