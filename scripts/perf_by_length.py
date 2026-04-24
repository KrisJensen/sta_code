"""
Here we test the ability of the STA to generate plans of different lengths for a few different noise levels.
"""

#%%

import pysta
from pysta import basedir
import pickle
import numpy as np
import torch
from pysta.maze_utils import compute_shortest_dists
np.random.seed(0)
torch.manual_seed(0)

#%%

lengths = [1,4, 6, 8,10,12,13,14,15]

noises = [
    {"rec_noise": 0.0, "adj_noise": 0.0, "adj_bias": -5e-3, "beta": 6},
    {"rec_noise": 1e-1, "adj_noise": 1e-2, "adj_bias": 0.0, "beta": 6},
    {"rec_noise": 2e-1, "adj_noise": 2e-2, "adj_bias": +5e-3, "beta": 6},
    {"rec_noise": 3e-1, "adj_noise": 3e-2, "adj_bias": +15e-3, "beta": 6},
]

def eval_representation(r, dists, goal, length):
    states = r.argmax(-1)
    good = 1
    state = states[0]
    assert dists[state, goal] == length
    
    for l in range(1, length+1):
        new_state = states[l]
        new_val = r[l, new_state]
        if new_val < 0.3 or dists[new_state, state] != 1:
            good = 0
        #print(state, new_state, new_val, good, goal)
        state = new_state
    #assert state == goal
    
    return good
        
#%%
iters = 20
goods = np.zeros((len(noises), iters, len(lengths)))

for inoise, noise in enumerate(noises):
    for iter_ in range(iters):
        
        print("\nIter", iter_, inoise)
        dists = np.zeros(1)
        while dists.max() < lengths[-1]:
            env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_rew = False, batch_size = len(lengths), dynamic_rew = False, max_steps = lengths[-1]+1, side_length = 6)
            dists = compute_shortest_dists(env.adjacency[0])
            #print(dists.max())
        
        goal = np.random.choice(np.where(dists >= lengths[-1])[0])
        
        env.goal = torch.zeros(env.batch, env.max_steps+1, dtype = int) + goal
        env.rews = torch.zeros(env.batch, env.max_steps+1, env.num_locs) + env.rew_nogoal # default negative reward
        for t in range(env.max_steps+1):
            env.rews[env.batch_inds, t, env.goal[:, t]] = env.rew_goal # positive reward at the goal location
        env.compute_value_function()

        sta = pysta.agents.SpaceTimeAttractor(env, tau = 50, iters_per_action = 1501, shift_time = 2.0, **noise)


        locs = []
        for length in lengths:
            locs.append(np.random.choice(np.where(dists[:, goal] == length)[0]))
        env.loc = env.loc*0+torch.tensor(locs)

        sta.step(env.observation())

        for ind in range(len(lengths)):
            r = sta.r[ind].detach().numpy()
            #plot_kwargs = {"walls": env.walls[0], "vmap": r, "loc": env.loc[ind], "goal": env.goal[0], "aspect": (1,1,6)}
            #pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = (8,5))
            goods[inoise, iter_, ind] = eval_representation(r, dists, goal, lengths[ind])
            print(lengths[ind], goods[inoise, iter_, ind])


#%%

results = {"lengths": lengths, "noises": noises, "goods": goods}
pickle.dump(results, open(f"{basedir}/data/comparisons/perf_by_length.p", "wb"))


# %% try with known goal distance

import time
t0 = time.time()

lengths = [1,5,10,15,20,25,30]
noise = {"rec_noise": 0.0, "adj_noise": 0.0, "adj_bias": 0.0, "beta": 20}
long_goods = np.zeros((iters, len(lengths)))


for iter_ in range(iters):
    dists = np.zeros(1)
    print("\nIter", iter_, "  t =", np.round((time.time()-t0)/60))
    while dists.max() < lengths[-1]+1:
        env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_rew = False, batch_size = len(lengths), dynamic_rew = False, max_steps = lengths[-1]+2, side_length = 10)
        dists = compute_shortest_dists(env.adjacency[0])
        #print(dists.max())

    goal = np.random.choice(np.where(dists >= lengths[-1]+1)[0])

    env.goal = torch.zeros(env.batch, env.max_steps+1, dtype = int) + goal
    env.rews = torch.zeros(env.batch, env.max_steps+1, env.num_locs) + env.rew_nogoal # default negative reward
    for ilength, length in enumerate(lengths):
        env.rews[ilength, length:, goal] = env.rew_goal # positive reward at the goal location only from the time we can reach it

    env.compute_value_function()

    sta = pysta.agents.SpaceTimeAttractor(env, tau = 35, iters_per_action = 15001, shift_time = 2.0, clip_minval = -1.0e2, **noise)

    locs = []
    for length in lengths:
        locs.append(np.random.choice(np.where(dists[:, goal] == length)[0]))
    env.loc = env.loc*0+torch.tensor(locs)

    sta.step(env.observation())

    for ind in range(len(lengths)):
        r = sta.r[ind].detach().numpy()
        # plot_kwargs = {"walls": env.walls[0], "vmap": r, "loc": None, "goal": None, "aspect": (1,1,21)}
        # pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = (35,10), filename = f"{basedir}/figures/temp/long{ind}.pdf")
        good = eval_representation(r, dists, goal, lengths[ind])
        long_goods[iter_, ind] = good
        print(lengths[ind], good)

results = {"lengths": lengths, "goods": long_goods}
pickle.dump(results, open(f"{basedir}/data/comparisons/perf_by_length_long.p", "wb"))

