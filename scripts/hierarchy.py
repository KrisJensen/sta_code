
#%%
import pysta
from pysta import basedir
import pickle
import numpy as np
import torch
from pysta.maze_utils import compute_shortest_dists, action_deltas, ind_a_ind
np.random.seed(0)
torch.manual_seed(0)

#%%

def walls_from_adj(adj):
    N = adj.shape[0]
    L = int(np.sqrt(N))
    walls = np.ones((N, 4))
    for s1 in range(N):
        for a in range(4):
            s2 = ind_a_ind(s1, a, L)
            if adj[s1, s2] and (s1 != s2):
                walls[s1, a] = 0.0
    return walls

def update_env(env, loc, goal):
    env.rews = torch.zeros(env.batch, env.max_steps+1, env.num_locs) + env.rew_nogoal # default negative reward
    for t in range(env.max_steps+1):
        env.rews[:, t, goal] = env.rew_goal # positive reward at the goal location
    env.compute_value_function()
    env.loc = env.loc*0+loc

#%%
max_steps = 8
noise = {"rec_noise": 0.0, "adj_noise": 0.0, "adj_bias": -5e-5}
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

abs2conc= [[np.concatenate([j*4+np.arange(4)+i*16 + k*4*16 for i in range(4)]) for j in range(4)] for k in range(4)]
abs2conc = np.array([u for v in abs2conc for u in v])

conc2abs = {}
for abs, conc in enumerate(abs2conc):
    for c in conc:
        conc2abs[c] = abs

# the small mazes that we'll construct a big one from
small_envs = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_maze = True, changing_trial_rew = False, batch_size = 16, dynamic_rew = False, max_steps = max_steps, side_length = 4, sample_wall_num = None)
small_adjs = small_envs.adjacency.numpy()

# how are the small mazes connected
abs_env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_maze = False, changing_trial_rew = False, batch_size = 1, dynamic_rew = False, max_steps = max_steps, side_length = 4, sample_wall_num = None)
abs_adj = abs_env.adjacency.numpy()[0]
abs_wall = abs_env.walls.numpy()[0]
abs_dists = compute_shortest_dists(abs_adj)
assert np.all(abs_wall == walls_from_adj(abs_adj))
abs_env.plot()

# construct big maze
big_env = pysta.envs.MazeEnv(rew_landscape = False, changing_trial_rew = False, batch_size = 1, dynamic_rew = False, max_steps = max_steps, side_length = 16)
big_adj = np.zeros((16*16, 16*16))

# set adjacency matrix appropriately for each sub-maze
for i1, inds in enumerate(abs2conc):
    big_adj[np.ix_(inds, inds)] = small_adjs[i1]

# now connect them appropriately
right_inds = np.arange(12,16) # local inds where I can go right
top_inds = np.array([3,7,11,15]) # local inds where I can go up
boundary_states = {i1: {i2: [] for i2 in range(16)} for i1 in range(16)}

# each location with no wall to the right
for i1 in np.where(abs_wall[:, 0] == 0)[0]:
    i2 = ind_a_ind(i1, 0, 4)
    num_locs = np.random.choice([1,2,3], p = [0.6, 0.3, 0.1])
    for s1 in np.random.choice(abs2conc[i1][right_inds], num_locs, replace = False): # pick random state that is on the right of the small maze
        s2 = big_env.nowall_neighbors[s1, 0] # what is adjacent state
        big_adj[s1, s2], big_adj[s2, s1] = 1.0, 1.0
        print(i1, s1, s2)
        
        boundary_states[i1][i2].append(s2)
        boundary_states[i2][i1].append(s1)

# each location with no wall above
for i1 in np.where(abs_wall[:, 2] == 0)[0]:
    i2 = ind_a_ind(i1, 2, 4)
    num_locs = np.random.choice([1,2,3], p = [0.6, 0.3, 0.1])
    for s1 in np.random.choice(abs2conc[i1][top_inds], num_locs, replace = False): # pick random state that is on top of the small maze
        s2 = big_env.nowall_neighbors[s1, 2] # what is adjacent state
        big_adj[s1, s2], big_adj[s2, s1] = 1.0, 1.0
        print(i1, s1, s2)

        boundary_states[i1][i2].append(s2)
        boundary_states[i2][i1].append(s1)

# add to environment and sanity check+plot
big_walls = walls_from_adj(big_adj)
big_env.sample_maze(walls = torch.tensor(big_walls[None, ...]))
assert np.all(big_env.adjacency[0].numpy() == big_adj)
big_dists = compute_shortest_dists(big_adj)
print(big_dists.max(), abs_dists.max())
start, goal = [int(x[0]) for x in np.where(big_dists == big_dists.max())]
goal = 143
start = 7
print(big_dists[start, goal])

#start, goal = 15, 150
big_env.plot(show = True, figsize = (6,6), lw = 7, filename = f"{basedir}/figures/temp/big_env.pdf", loc = start, goal = goal)

#%% run abstract STA

abs_start, abs_goal = [conc2abs[s] for s in [start, goal]]
print(abs_dists[abs_start, abs_goal])

iters_per_action = 1000
abs_sta = pysta.agents.SpaceTimeAttractor(abs_env,  beta = 9.0, tau = 50, iters_per_action = iters_per_action)
update_env(abs_env, abs_start, abs_goal)
action = abs_sta.step(abs_env.observation())
abs_r = abs_sta.r[0].numpy()

plot_kwargs = {"walls": abs_env.walls[0], "vmap": abs_r, "loc": abs_start, "goal": abs_goal, "aspect": (1,1,5.5)}
pysta.plot_utils.plot_perspective_attractor(**plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = (8,3))

abs_env.plot(loc = abs_start, goal = abs_goal)
abs_next = np.argmax(abs_r[1])

big_goal = np.array(boundary_states[abs_start][abs_next])
print("dist to next abs:", big_dists[start, big_goal].min())

#%% run primitive STA

big_sta = pysta.agents.SpaceTimeAttractor(big_env,  beta = 9.0, loc_strength = 20, tau = 50, iters_per_action = iters_per_action, **noise)
update_env(big_env, start, big_goal)
big_sta.step(big_env.observation())

#%% extract and save info

# raw activity
big_r = big_sta.r[0].numpy()

# abstract activity
abs_r_big = np.zeros(big_r.shape)
for t in range(max_steps):
    for s in range(16):
        abs_r_big[t, abs2conc[s]] = abs_r[t, s]

# save result
results = {"walls": big_walls, "abs_r": abs_r_big, "big_r": big_r, "start": start, "goal": goal}
pickle.dump(results, open(f"{basedir}/data/comparisons/hierarchy.p", "wb"))


# %%
