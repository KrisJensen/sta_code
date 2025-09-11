
#%%
import pickle
import pysta
pysta.reload()
import os
os.chdir(f"{pysta.basedir}/scripts")
from submit_slurm import submit_slurm

seeds = [21, 22, 23, 24, 25]
base_command = f"python {pysta.basedir}/pysta/train_rnn.py --overwrite 1"
commands = {}

#%% train WM RNNs
commands["WM"] = []
for seed in seeds:
    command = f"{base_command} --seed {seed}"
    print(submit_slurm(command, f"train_RNN_WM_{seed}"))
    commands["WM"].append(command)
    
#%% train WM changing maze
commands["changing_maze"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_{seed}"))
    commands["changing_maze"].append(command)
    
#%% train WM changing maze
commands["changing_maze_long"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --num_epochs 250000 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_long_{seed}", time = "48:00:00"))
    commands["changing_maze_long"].append(command)
    
    
#%% train WM changing maze old
commands["changing_maze_old"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix old_ --r_reg 1e-4 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_old_{seed}"))
    commands["changing_maze_old"].append(command)
    
    
#%% train WM changing maze low_reg
commands["changing_maze_lowreg"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix lowreg_ --r_reg 1e-6 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_lowreg_{seed}"))
    commands["changing_maze_lowreg"].append(command)

    
#%% train WM changing maze mid_reg
commands["changing_maze_midreg"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix midreg_ --r_reg 5e-6 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_midreg_{seed}"))
    commands["changing_maze_midreg"].append(command)
    
    
    
#%% train WM changing maze low_Wreg
commands["changing_maze_lowWreg"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix lowWreg_ --W_reg 2e-7 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_lowWreg_{seed}"))
    commands["changing_maze_lowWreg"].append(command)
    
    

#%% train WM changing maze high_reg
commands["changing_maze_highreg"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix highreg_ --r_reg 1e-4 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_highreg_{seed}"))
    commands["changing_maze_highreg"].append(command)
    
    
#%% train WM changing maze low_rate
commands["changing_maze_lrate"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix lrate_ --lrate 1e-4 --num_epochs 150000 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_lowrate_{seed}"))
    commands["changing_maze_lrate"].append(command)
    
#%% train WM changing maze high_rate
commands["changing_maze_hrate"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --prefix hrate_ --lrate 5e-4 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_highrate_{seed}"))
    commands["changing_maze_hrate"].append(command)
    
    
    
#%% train WM changing maze old seeds
commands["changing_maze_seeds"] = []
for seed in [11,12,13,14,15]:
    command = f"{base_command} --changing_trial_maze 1 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_{seed}"))
    commands["changing_maze_seeds"].append(command)
    
    
#%% train RNNs with continual reward input
commands["relrew"] = []
for seed in seeds:
    command = f"{base_command} --iters_per_action 9 10 11 --working_memory 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_relrew_{seed}"))
    commands["relrew"].append(command)

#%% train egocentric
commands["egocentric"] = []
for seed in seeds:
    #command = f"{base_command} --output_format egocentric --nonlin_output 1 --seed {seed}"
    command = f"{base_command} --output_format egocentric --seed {seed}"
    print(submit_slurm(command, f"train_RNN_egocentric_{seed}"))
    commands["egocentric"].append(command)

#%% then train on simpler tasks
commands["moving_goal"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_moving_goal_{seed}"))
    commands["moving_goal"].append(command)
    
commands["static_goal"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --dynamic_rew 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_static_goal_{seed}"))
    commands["static_goal"].append(command)
    
#%% then train on simpler tasks with relative reward
commands["moving_goal_rel"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --iters_per_action 9 10 11 --working_memory 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_moving_goal_rel_{seed}"))
    commands["moving_goal_rel"].append(command)
    
commands["static_goal_rel"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --dynamic_rew 0 --iters_per_action 9 10 11 --working_memory 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_static_goal_rel_{seed}"))
    commands["static_goal_rel"].append(command)

#%% train networks of different sizes 100-10000
sizes = [100,200,300,400,600,800,1000]
commands["sizes"] = []
seed = 20
for size in sizes:
    command = f"{base_command} --Nrec {size} --seed {seed}"
    print(submit_slurm(command, f"train_RNN_size{size}_{seed}", time = "32:00:00"))
    commands["sizes"].append(command)    

#%% store commands

pickle.dump(commands, open(f"{pysta.basedir}/slurm/commands/train_all_rnns.p", "wb"))


