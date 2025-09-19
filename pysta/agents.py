
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%% base agent
class BaseAgent(nn.Module):
    classname = "BaseAgent"
    label = "base"

    def __init__(self, env, rec_noise = 1e-3, ent_reg = 1e-5, greedy = False, force_optimal = False, iters_per_action = 1, tau = 1, **kwargs):
        """
        RNN agent that learns to navigate in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        rec_noise : float
            Noise added to the recurrent dynamics
        ent_reg: float
            Amount of entropy regularization to add
        greedy : bool
            If True, take greedy actions, otherwise sample from the policy
        force_optimal : bool
            Only allow optimal actions. If there are multiple actions, renormalize the policy over these.
        iters_per_action : int or list of ints
            Number of RNN iterations to run for each step of the environment.
            This get's multiplied with 'env.initial_iters' for the true number of planning RNN iterations.
        tau : float
            timescale of the dynamics. r_{t+1} = (1-1/tau) * r_t + (1/tau) * f(r_t, x_{t+1})
        """
        
        # pytorch boilerplate
        super(BaseAgent, self).__init__()

        # store some hyperparameters
        self.env = env
        self.greedy = greedy
        self.force_optimal = force_optimal
        self.iters_per_action = iters_per_action
        self.tau = tau
        self.rec_noise = rec_noise
        self.store_all_activity = False
        self.ent_reg = ent_reg
    
        # and get some from the environment
        self.Nout = self.env.output_dim # dimensionality of the policy we're learning
        self.Nin = self.env.obs_dim # dimensionality of the observations

        
        # instantiate parameters
        self.initialise_weights()
        
        # reset the model to its initial conditions
        self.reset()
        
        return
        
        
    def initialise_weights(self):
        """
        Instantiate the model parameters
        """
        raise NotImplementedError
    
    def allo_to_ego_pi(self, allo_pi):
        """
        convert an allocentric policy to an egocentric policy by renormalizing over neighboring states
        in this version, each action is given the non-normalized probability associated with the result state (instead of distributing the probability mass)
        This is to ensure that the greedy actions agree
        """
        
        ego_pi = torch.zeros(allo_pi.shape[0], self.env.num_actions) # new egocentric policy
        new_locs = self.env.neighbors[self.env.batch_inds, self.env.loc, :] # for each action, where do I end up
        for a in range(self.env.num_actions): # for each action
            new_locs_a = new_locs[:, a] # where would I end up
            ego_pi[:, a] = allo_pi[self.env.batch_inds, new_locs_a] # what is the probability of going here
        return ego_pi / ego_pi.sum(-1, keepdims = True)
        

    @property
    def name(self):
        """generates a string representation of the agent"""
        if type(self.iters_per_action) in [int, np.int32, np.int64]:
            iter_str = f"iter{self.iters_per_action}"
        else:
            iter_str = f"iter{'-'.join([str(val) for val in self.iters_per_action])}"

        tau_str = f"tau{self.tau}"
        force_str = "opt" if self.force_optimal else "agent"
        #return f"{self.classname}/{iter_str}/{tau_str}/{force_str}"
        return f"{self.classname}/{iter_str}_{tau_str}_{force_str}"

    def reset(self):
        """
        reset the agent.
        sets the initial condition and empties caches storing data.
        """
        
        # no gradients for setting environment
        with torch.no_grad():
            self.env.reset() # reset environment state

            # also instantiate some lists to store data along the way
            self.store = [] # store many things
            self.all_acts = [[], [], []] # also store in between environment update steps
            
        # need gradients for setting initial z
        self.z = torch.zeros(torch.Size([self.env.batch])+self.z0.shape, device = self.z0.device) + self.z0[None, ...]
        self.r = self.phi(self.z)
        
        # instantiate loss functions
        self.acc_loss = torch.tensor(0.0) # accuracy
        self.weight_loss = self.env.batch * self.calc_parameter_reg() # parameter regularization
        self.rate_loss = self.calc_activity_reg() # rate regularization
        self.ent_loss = torch.tensor(0.0) # entropy regularization
        self.update_optimal_actions()
        
        return
    
    def sample_action(self):
        """
        Samples an action for each trial in a batch from the current policy.
        Note that 'self.action' is the actual chosen action.
        'self.env_action' is the action passed to the environment and can be different if we enforce optimality.
        We distinguish between the two because e.g. accuracies are still computed from the sampled action.

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch

        """
        
        with torch.no_grad():
            if self.env.output_format == "allocentric":
                adj = self.env.adjacency[self.env.batch_inds, self.env.loc, :] # renormalize over adjacent states
                assert adj.sum(-1).min() >= 2 # should have adjacent states
                pi = (self.pi+1e-20) * adj # add some jitter to make sure policy is not exactly zero
                pi = pi/pi.sum(-1, keepdims = True)
            else:
                pi = self.pi # already in egocentric space, everything is possible
                
            if self.greedy: # pick the most likely action
                self.action = torch.argmax(pi, -1)
            else: # sample an action
                try:
                    self.action = torch.multinomial(pi, 1)[..., 0]
                except RuntimeError:
                    print(pi.min(), pi.max(), self.pi.min(), self.pi.max(), self.logpi.min(), self.logpi.max())

                    pickle.dump(self, open("./temp.p", "wb"))
                    raise Error
                
            if self.force_optimal: # optionally renormalize over optimal actions first
                # add small epsilon in case there are no optimal actions -> uniform policy
                opt_pis = pi * self.optimal_actions + torch.rand(pi.shape)*1e-20
                if self.greedy: # pick most likely action
                    self.env_action = torch.argmax(opt_pis, -1)
                else: # sample an action
                    self.env_action = torch.multinomial(opt_pis, 1)[..., 0]
            else: # if we're not enforcing optimality, the action passed to environment is just the original action
                self.env_action = self.action
            
        return self.action
    
    def step(self, observation):
        """
        Perform one 'update step'
        Here, an update step corresponds to all the computations happening for one environment step.
        This can include multiple iterations of recurrent network dynamics.

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """
        
        raise NotImplementedError
    
    def calc_activity_reg(self, not_finished = None):
        """
        Compute firing rate regularization loss. 0 by default.
        """
        return 0.0
    
    def calc_parameter_reg(self):
        """
        Compute weight regularization loss. 0 by default.
        """
        return 0.0

    def update_loss(self):
        """
        Accumulate losses.
        Treat accuracy loss, rate loss, and parameter loss separately.
        Losses are only added for trials that have not finished.
        """
        
        not_finished = torch.where(~self.env.finished)[0] # trials that have not finished 
        
        if self.env.step_num >= 0: # only apply prediction and parameter loss during execution phase
            
            # policy loss (sum_{a \in opt_as} pi(a))
            opt_probs = (self.pi*self.optimal_actions)[not_finished, :].sum(-1)
            assert opt_probs.max() < 1.0 + 1e-5
            self.acc_loss = self.acc_loss + (1.0 - opt_probs).sum()
            
            # entropy loss
            jitter = 1e-5 # add a little bit of jitter to avoid nans
            pi_ent = (self.pi + jitter) / (1+self.pi.shape[-1] * jitter)
            self.ent_loss = self.ent_loss + self.ent_reg * (pi_ent*pi_ent.log())[not_finished, :].sum() # want to maximize entropy; minimize -H = E[pi logpi]
    
        return
    
    def update_store(self):
        """
        Update list of environment/agent states with the current state.
        """
        
        corrects = self.optimal_actions[self.env.batch_inds, self.action]
        corrects[self.env.finished] = torch.nan

        self.store.append({
        "rs": self.r.detach(),
        "zs": self.z.detach(),
        "loc": self.env.loc,
        "action": self.action,
        "env_action": self.env_action,
        "optimal_actions": self.optimal_actions,
        "step_num": self.env.step_num,
        "finished": self.env.finished.clone(),
        "pi": self.pi.detach(),
        "xs": self.env.observation(),
        "corrects" : corrects
        })
        
    def update_optimal_actions(self):
        """
        cache optimal actions
        """
        with torch.no_grad():
            self.optimal_actions = self.env.optimal_actions() # tensor ((batch, output_dim)

    def forward(self, store = False):
        """
        Run a single batch of trials

        Parameters
        ----------
        store : bool
            if true, store environment and agent states after every action
            
        Returns
        ----------
        avg_loss : tensor
            average loss across trials in a batch
        """
        
        self.reset() # reset agent
        
        while not torch.all(self.env.finished): # as long as there are some trials left to act in
            self.update_optimal_actions() # cache the optimal actions at the current location
            
            x = self.env.observation().to(self.z0.device) # observation at this point in time
            self.step(x) # update RNN state, compute policy, and sample an action
            self.update_loss() # update performance loss 
            
            # now update environment and optionally store env+agent state (don't propagate gradients through this)
            with torch.no_grad():
                if store:
                    self.update_store()
                self.env.step(self.env_action) # action passed to the environment (optionally restricted to be optimal)
        
        # loss is combined accuracy and regularization losses, normalized by the batch size
        return (self.acc_loss + self.weight_loss + self.rate_loss + self.ent_loss) / self.env.batch

    def eval(self, num_eval = 5):
        """
        Compute average loss and accuracy across a number of trials

        Parameters
        ----------
        n_eval : int
            number of batches to average over
            
        Returns
        ----------
        avg_loss : float
            average loss across trials in all batches
        avg_acc : float
            average accuracy across trials in all batches
        """
        losses, accs = [], [] # lists to concatenate across batches
        for _ in range(num_eval): # for each batch
            # simulate the trials
            loss = self.forward(store = True).detach().cpu().numpy()
            losses.append(loss) # append loss
            corrects = torch.stack([s["corrects"] for s in self.store if s["step_num"]>=0]) # accuracy for each trial and step in execution phase
            accs.append(torch.nanmean(corrects, axis = 0).mean()) # average
            
        return np.mean(losses), np.mean(accs)
            
    def plot_trial(self, filename = None, run_trial = True, trial_num = 0, values = True, vmap = None, cmap = "coolwarm"):
        """
        Plot a summary plot for a trials

        Parameters
        ----------
        filename : str
            filename to save to
        run_trial : bool
            whether to run a new trial (True) or use cached information (False)
        trial_num : int
            which trial number within the batch to plot
        values : bool
            for the reward landscape task, whether to plot a value map (True) or reward map (False)
        vmap : tensor
            optional data to plot as a heatmap for each time point
        cmap : str
            colormap to use
        """
        
        if run_trial: # optionally simulate a new trial
            self.forward(store = True)
        
        # find out which time points have not finished
        fs = 1 - np.array([store["finished"][trial_num].numpy() for store in self.store])
        T = min(np.sum(fs)+1, len(self.store)) # number of time points to plot
        fig, axs = plt.subplots(1, T, figsize = (2*T, 2))
        for t, ax in enumerate(axs): # for each time point
            data = self.store[t] # get data for this time point
            # plot a panel
            vmap_t = None if vmap is None else vmap[t] # optionally plot some useful data
            self.env.plot(loc = data["loc"][trial_num], step_num = data["step_num"], ax = ax, trial_num = trial_num, values = values, cmap = cmap, vmap = vmap_t)
        plt.tight_layout = True
        if filename is not None:
            plt.savefig(filename, bbox_inches = "tight")
        plt.close()
        
        return


#%% Vanilla RNN
class VanillaRNN(BaseAgent):
    classname = "VanillaRNN"
    label = "rnn"
    
    def __init__(self, env, Nrec = 800, W_reg = 1e-3, r_reg = 1e-3, nonlin_output = False, **kwargs):
        """
        RNN agent that learns to navigate in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        Nrec : int
            Number of relu units in the RNN's hidden layer
        W_reg : float
            Regularization strength for the parameters
        r_reg : float
            Regularization strength for the hidden layer activity
        nonlin_output : bool
            If True, have an additional hidden layer in the readout. Otherwise use a linear readout from the hidden state.
        """
    
        # store some model-specific parameters
        self.Nrec = Nrec
        self.nonlin_output = nonlin_output
        self.W_reg, self.r_reg = W_reg, r_reg 
        self.phi = F.relu # use a ReLU nonlinearity
        
        # initialise BaseAgent
        super(VanillaRNN, self).__init__(env, **kwargs)
        
    @property
    def name(self):
        """generates a string representation of the agent"""
        nonlin_str = "nonlinout" if self.nonlin_output else "linout"
        basename = super(VanillaRNN, self).name
        return f"{basename}/N{self.Nrec}_{nonlin_str}"
    
    def initialise_weights(self):
        """
        Instantiate the learnable model parameters
        """

        self.z0 = nn.Parameter(torch.randn(self.Nrec, 1), requires_grad=True) # RNN initial condition
        
        # recurrent weight matrix
        self.Wrec = nn.Parameter(torch.randn(self.Nrec, self.Nrec) / np.sqrt(self.Nrec), requires_grad=True)
        # input weight matrix
        self.Win = nn.Parameter(torch.randn(self.Nrec, self.Nin) / np.sqrt(self.Nin), requires_grad=True)
        # hidden state bias
        self.brec = nn.Parameter(torch.zeros(self.Nrec, 1), requires_grad=True)
        
        # now initial the output function, which can be either nonlinear or linear
        if self.nonlin_output:
            # create one hidden layer between the RNN and policy

            # weights and bias to hidden output layer
            self.Wout1 = nn.Parameter(torch.randn(int(np.round(self.Nrec/2)), self.Nrec) / np.sqrt(self.Nrec), requires_grad=True)
            self.bout1 = nn.Parameter(torch.zeros(self.Wout1.shape[0], 1), requires_grad=True)

            # weights and bias to policy
            self.Wout = nn.Parameter(torch.randn(self.Nout, self.Wout1.shape[0]) / np.sqrt(self.Wout1.shape[0]), requires_grad=True)
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)

        else:
            # just a single weight and bias for a linear readout to out policy
            self.Wout = nn.Parameter(torch.randn(self.Nout, self.Nrec) / np.sqrt(self.Nrec), requires_grad=True)
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)
        
        return

        
    def step(self, observation):

        batch = observation.shape[0] # batch size
        
        network_iters = self.iters_per_action if type(self.iters_per_action) in [int, np.int32, np.int64] else np.random.choice(self.iters_per_action)
        for _ in range(network_iters): # optionally several RNN steps per action
            
            # recurrent noise
            rec_noise = torch.randn(batch, self.Nrec, 1, device = self.z0.device) * self.rec_noise
            
            # feedforward input
            ff_inp = self.Win @ observation[..., None]
            
            # recurrent input
            rec_inp =  self.Wrec @ self.r

            # update neuron potentials
            self.z = (1 - 1/self.tau) * self.z + (1/self.tau)*(rec_inp + ff_inp + self.brec + rec_noise)
            
            # compute firing rates
            self.r = self.phi(self.z) # firing rate
            
            # update rate loss for trials that have not finished
            self.rate_loss = self.rate_loss + self.calc_activity_reg(torch.where(~self.env.finished)[0])
            
            if self.store_all_activity: # store activity at every RNN iteration
                self.all_acts[0].append(self.r.detach().numpy())
                self.all_acts[1].append( self.env.loc.detach().numpy())
                self.all_acts[2].append(self.env.step_num)
        
        # compute output
        if self.nonlin_output:
            rout = self.phi(self.Wout1 @ self.r + self.bout1) # pass through hidden layer
        else:
            rout = self.r # just linear readout
        self.logpi = (self.Wout @ rout + self.bout)[..., 0] # flatten for each batch (batch x actions x 1) -> (batch, actions)

        # normalize log policy
        self.logpi = self.logpi - self.logpi.logsumexp(-1, keepdims = True)
        self.pi = self.logpi.exp() # compute policy (batch, actions).
        
        # sample an action from the policy
        self.action = self.sample_action()
        
        return self.action
    
    def calc_activity_reg(self, not_finished = None):
        """
        Compute firing rate regularization loss

        Parameters
        ----------
        not_finished : bool
            trials within the batch which have not yet finished.
            if None: assume no trials are finished.

        Returns
        ----------
        reg_loss : tensor
            firing rate regularization loss. Summed across trials and neurons
        """
        
        if not_finished is None:
            reg_loss = self.r_reg * (self.r**2).sum() # magnitude of hidden state vector
        else:
            reg_loss = self.r_reg * (self.r[not_finished]**2).sum() # magnitude of hidden state vector
        return reg_loss
    
    def calc_parameter_reg(self):
        """
        Compute weight regularization loss
            
        Returns
        ----------
        reg_loss : tensor
            weight regularization loss. sum across all parameters.
        """
        
        reg_loss = self.W_reg * torch.stack([torch.square(p).sum() for p in self.parameters()]).sum() # magnitude of total weight vecto

        return reg_loss

#%% spacetime attractor


class SpaceTimeAttractor(BaseAgent):
    classname = "SpaceTimeAttractor"
    label = "sta"
    
    def __init__(self, env, beta = 9.0, tau = 50, iters_per_action = 400, shift_time = 2.0, rec_noise = 1e-1, adj_noise = 1e-2, **kwargs):
        """
        Spacetime attractor that optimises a reward function

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        beta : float
            temperature parameter for the reward function. Performance is somewhat sensitive to this
        """
        
        self.num_modules = env.max_steps+1 # current loc plus max_steps future locs
        self.num_locs = env.num_locs
        self.batch = env.batch
        self.adj_noise = adj_noise
        self.shift_time = shift_time
        
        self.Nrec = self.num_modules * self.num_locs # total number of neurons across modules
        self.beta = beta # reward function temperature parameters
        self.exact = False # perform exact inference
        
        # initialise BaseAgent
        kwargs["tau"] = tau
        kwargs["iters_per_action"] = iters_per_action
        super(SpaceTimeAttractor, self).__init__(env, **kwargs)
        self.rec_noise = rec_noise # recurrent noise to ensure robustness
    
    def phi(self, x):
        """
        for now just assume that z is log and r is exp
        """

        return x.exp()
    
    def initialise_weights(self):
        """
        Instantiate the handcrafted model parameters
        """

        A = self.env.adjacency.clone()
        adj_noise = self.adj_noise
        
        self.Wrec_fwd, self.Wrec_bwd, self.Wrec_self = [torch.zeros(self.batch, self.Nrec, self.Nrec) for _ in range(3)]
        self.Win = torch.zeros(self.Nrec, self.Nin)
        self.Wout = torch.zeros(self.env.num_locs, self.Nrec) # inherently allocentric; can convert to egocentric subsequently
        self.Wrec_shift = torch.zeros(self.Nrec, self.Nrec)
        bias = -5e-3
        
        # recurrent weights
        for mod1 in range(self.num_modules-1):
            mod1_inds, mod2_inds = [torch.arange(self.num_locs)+self.num_locs*ind for ind in [mod1, mod1+1]]
            self.Wrec_self[:, mod1_inds, mod1_inds] = 1 # self connections
            self.Wrec_shift[mod1_inds, mod2_inds] = torch.ones(self.num_locs) # shift connections to next module
            #print(mod1_inds, mod2_inds)
            for i1, ind1 in enumerate(mod1_inds):
                for i2, ind2 in enumerate(mod2_inds):
                    self.Wrec_fwd[:, ind2, ind1] = A[:, i2, i1].clone() + (torch.rand(self.batch) - 2.0)*adj_noise + bias
                    self.Wrec_bwd[:, ind1, ind2] = A[:, i1, i2].clone() + (torch.rand(self.batch) - 2.0)*adj_noise + bias
        
        self.Wrec = self.Wrec_fwd + self.Wrec_bwd
        
        # input weights
        self.Win[:self.num_locs, :self.num_locs] = torch.eye(self.num_locs) # current loc
        self.Win[self.num_locs:, 2*self.num_locs:-2*self.num_locs] = torch.eye(self.Win.shape[0] - self.num_locs) # reward
        
        self.Wout[:, self.num_locs:2*self.num_locs] = torch.eye(self.num_locs)
        
        # uniform in space at each time
        self.z0 = torch.log(torch.ones(self.num_modules, self.num_locs) / self.num_locs)
        
        #bias is zero for this agent 
        self.brec = torch.zeros(self.Nrec, 1)
        
        return

    def calc_policy(self):
        """compute policy from firing rate"""
        self.pi = (self.Wout @ self.r.reshape(self.batch, -1, 1))[..., 0] # policy ends up being activity in second module
        if self.env.output_format == "egocentric": # convert to an egocentric policy
            self.pi = self.allo_to_ego_pi(self.pi)

    def obs_to_rew_func(self, observation, flat = False):
        obs_mod = observation.reshape(self.batch, -1, self.num_locs) # module by module
        rew_inds = torch.cat([torch.arange(1), torch.arange(2, obs_mod.shape[1]-2)])
        logrews = obs_mod[:, rew_inds, :] #.clone() # reward function (batch, modules, locs)
        logrews[:, 0, :] *= 20.0 # very strong signal to start at current loc
        logrews *= self.beta # multiply by temperature parameter
        logrews -= logrews.logsumexp(axis = -1, keepdims = True) # turn into a distribution over locations and add final vector dimension 
        if flat:
            obs_mod[:, rew_inds, :] = logrews # replace reward function in observation
            return obs_mod.reshape(self.batch, -1, 1)
        else:
            return logrews[..., None]

    def step(self, observation):

        clip_minval = torch.tensor(-100) # minimum value of logps

        # now extract reward function from observation
        logrews = self.obs_to_rew_func(observation.clone(), flat = True) # flatten for each trial (batch, modules*locs)

        network_iters = self.iters_per_action if type(self.iters_per_action) in [int, np.int32, np.int64] else np.random.choice(self.iters_per_action)
        
        for iter_ in range(network_iters):

            rs = self.r.reshape(self.batch, -1, 1) # firing rate but add vector dim
            dzdt = (self.Win @ logrews + torch.maximum(clip_minval.exp(), self.Wrec_fwd @ rs).log() + torch.maximum(clip_minval.exp(), self.Wrec_bwd @ rs).log())

            if (iter_ < self.tau*self.shift_time) and (self.env.step_num > 0):
                #print(rs.shape, self.Wrec_shift.shape, dzdt.shape)
                dzdt += torch.maximum(clip_minval.exp(), self.Wrec_shift @ rs).log()
                
            dzdt = dzdt.reshape(self.batch, -1, self.num_locs)
            bias = self.brec.reshape(1, -1, self.num_locs)

            # recurrent noise
            rec_noise = torch.randn(dzdt.shape, device = self.z0.device) * self.rec_noise

            # update activity
            self.z = (1-1/self.tau) * self.z + 1/self.tau * (dzdt + rec_noise + bias) # update activity
            
            # normalize and threshold
            self.z = self.z - self.z.logsumexp(axis = -1, keepdims = True)
            self.z = torch.clip(self.z, min = clip_minval, max = 0.0)
            self.r = self.phi(self.z) # apply exp nonlinearity
            
            #print(self.r.sum(-1).mean(), self.r.sum(-1).std())
            
            if self.store_all_activity: # store activity at every RNN iteration
                self.all_acts[0].append(self.r.detach().numpy())
                self.all_acts[1].append( self.env.loc.detach().numpy())
                self.all_acts[2].append(self.env.step_num)

        self.calc_policy()
        # sample an action from the policy
        self.action = self.sample_action()
        
        return self.action
            
    def reset(self):
        """also need to precompute successor matrix"""
        super(SpaceTimeAttractor, self).reset()
        self.initialise_weights() # reinitialize weights
        
        return

    def plot_representation(self, filename = None, trial_num = 0, **kwargs):
        """
        Plot a summary plot for a trials

        Parameters
        ----------
        filename : str
            filename to save to
        trial_num : int
            which trial number within the batch to plot
        """

        # find out which time points have not finished
        fig, axs = plt.subplots(1, self.num_modules, figsize = (2*self.num_modules, 2))
        for t, ax in enumerate(axs): # for each time point, plot a panel
            loc = self.env.loc[trial_num] if t == 0 else torch.argmax(self.r[trial_num, t]) # predicted location
            self.env.plot(loc = loc, vmap = self.r[trial_num, t, :], step_num = t, ax = ax, trial_num = trial_num,
                          cmap  = "YlOrRd", plot_optimal_actions = t<len(axs)-1, vmin = -0.1, vmax = 1.2, **kwargs)
        plt.tight_layout = True
        if filename is not None:
            plt.savefig(filename, bbox_inches = "tight")
        plt.close()
        
        return

    
#%% Successor representation agent
class SRLearner(BaseAgent):
    classname = "SRLearner"
    label = "sr"

    def __init__(self, env, gamma = 0.95, beta = 5.0, **kwargs):
        """
        Successor representation agent that navigates in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        gamma : the discount factor used to compute the successor matrix
        """
        
        # instantiate super class
        self.gamma = gamma
        self.beta = beta # temperature parameter
        super(SRLearner, self).__init__(env, **kwargs)
        self.Nrec = 2*self.env.num_locs # expected occupancy of each location and expected value of each location
        
        return
    
    def phi(self, x):
        return x
    
    def reset(self):
        """also need to precompute successor matrix"""
        super(SRLearner, self).reset()
        adj = self.env.adjacency # (batch, to, from)
        T = adj / adj.sum(1, keepdims = True) # transition matrix
        self.M = torch.linalg.inv(torch.eye(T.shape[-1])[None, ...] - self.gamma*T)
        self.r = None
        
        return
    
    def initialise_weights(self):
        """
        Model parameters are empty; we will compute analytically
        """
        self.z0 = torch.zeros(self.env.num_locs) # z is value, which we initialize to zero
        return

    def step(self, observation):
        """
        Perform one 'update step'
        For simplicity, we just multiply the average occupancy with the average reward function.
        If the reward format is 'relative', this ends up being the 'reward-to-go'.
        in the future, we could think about also using the 'occupancy-to-go' instead of just exponentially decayed occupancy.

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """

        rew_func = observation.reshape(self.env.batch, -1, self.env.num_locs)[:, 1:-2, :].clone() # reward function (batch, steps, locs)
        # take an average across time

        mean_rew = rew_func[:, 1:, :].mean(1, keepdims = True) # ignore rew at current time
        # compute value function
        self.values = (mean_rew @ self.M)[:, 0, :] # (batch, locs)
        self.z = self.values
        exp_occ = self.M[self.env.batch_inds, :, self.env.loc]
        self.r = torch.cat([self.values, exp_occ], 1)[..., None]
        
        self.pi = (self.beta * self.values).exp() # allocentric policy
        self.pi /= self.pi.sum(-1, keepdims = True) # normalize
        
        if self.env.output_format == "egocentric":
            self.pi = self.allo_to_ego_pi(self.pi) # convert to egocentric

        self.action = self.sample_action() # sample an action from the policy
        
        return self.action
    
    def plot_trial_values(self, filename = None, run_trial = True, trial_num = 0, cmap = "coolwarm"):
        
        if run_trial: # optionally simulate a new trial
            self.forward(store = True)
        vmap = [s["zs"][trial_num].flatten() for s in self.store]
            
        self.plot_trial(filename = filename, run_trial = False, trial_num = trial_num, vmap = vmap, cmap = cmap)

#%% TD learning agent

class TDLearner(BaseAgent):
    classname = "TDLearner"
    label = "td"

    def __init__(self, env, beta = 5.0, tau = 20, **kwargs):
        """
        Successor representation agent that navigates in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        gamma : the discount factor used to compute the successor matrix
        tau : the inverse of the learning rate
        """
        
        # instantiate super class
        self.beta = beta # temperature parameter
        kwargs["tau"] = tau
        super(TDLearner, self).__init__(env, **kwargs)
        
        return
    
    def phi(self, x):
        return x
    
    def reset(self):
        """also need to store initial location"""
        super(TDLearner, self).reset()
        self.prev_loc = self.env.loc
        self.prev_not_finished = torch.ones(self.env.num_locs)
        return
    
    def initialise_weights(self):
        """
        Model parameters are empty; we will compute analytically
        """
        self.values = torch.zeros(self.env.num_locs)+1.0 # optimistic value initialisation
        self.z0 = self.values.clone() # our firing rates will also be the values
        return
    
    def run_td_update(self):
        new_rew = self.env.latest_rew # the reward we just got
        #print(self.env.step_num, new_rew)
        prev_V = self.values[self.prev_loc]
        new_V = self.values[self.env.loc] * (~self.env.finished).to(float) # no more value for finished episodes
        
        self.td_error = (new_rew + new_V - prev_V) * self.prev_not_finished  # TD error (set to zero for finished episodes)
        
        #raise NotImplementedError # check that this works correctly with batched data
        prev_locs_1hot = F.one_hot(self.prev_loc, num_classes = self.env.num_locs) # (batch, num_locs)
        updates = (prev_locs_1hot * self.td_error[:, None]).sum(0) / self.prev_not_finished.sum() # avg over all td errors for each location in the batch
        self.values += (1/self.tau)*updates

    def step(self, observation):
        """
        Perform one 'update step'
        This involves both choosing an action and updating our value function

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """

        # first select an action
        self.pi = (self.beta * self.values).exp()[None, ...] + torch.zeros(observation.shape[0], self.env.num_locs) # allocentric policy is just value
        self.pi /= self.pi.sum(-1, keepdims = True) # normalize
        if self.env.output_format == "egocentric":
            self.pi = self.allo_to_ego_pi(self.pi) # convert to egocentric

        self.action = self.sample_action() # sample an action from the policy
        
        # now update our values based on previous experience
        if self.env.step_num >= 1:
            self.run_td_update()
            
        self.z = self.values[None, :] + torch.zeros(self.env.batch, self.env.num_locs) # broadcast to batch size
        self.r = self.phi(self.z)
        self.prev_loc = self.env.loc
        self.prev_not_finished = (~self.env.finished).to(float)
        
        return self.action
    
    def forward(self, store = False):
        loss = super(TDLearner, self).forward(store = store) # run default forward pass
        self.run_td_update() # important to update values based on final sample as well
        return loss
    
    def plot_trial_values(self, filename = None, run_trial = True, trial_num = 0, cmap = "YlOrRd"):
        
        if run_trial: # optionally simulate a new trial
            self.forward(store = True)
        vmap = [s["zs"][trial_num].flatten() for s in self.store]
            
        self.plot_trial(filename = filename, run_trial = False, trial_num = trial_num, vmap = vmap, cmap = cmap)



#%% Dynamic programming agent

class DPAgent(BaseAgent):
    classname = "DPAgent"
    label = "dp"
    
    def __init__(self, env, beta = 5.0, **kwargs):
        """
        Dynamic programming agent that just computes an optimal value function once and for all

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        """
        
        self.beta = beta
        self.Nrec = env.num_locs + env.max_steps + np.prod(env.vs[0].shape)
        
        # initialise BaseAgent
        super(DPAgent, self).__init__(env, **kwargs)
        
        return
    
    def phi(self, x):
        return x
    
    def initialise_weights(self):
        self.z0 = torch.zeros(self.Nrec, 1)
    
    def update_z_and_r(self):
        self.t = torch.tensor(max(0, self.env.step_num))
        self.flat_t = F.one_hot(self.t, num_classes = self.env.max_steps)+torch.zeros(self.env.batch, self.env.max_steps)
        self.loc = F.one_hot(self.env.loc, num_classes = self.env.num_locs)
        
        self.values = self.env.vs
        self.z = torch.cat([self.values.reshape(self.env.batch, -1), self.loc, self.flat_t], dim = -1)[..., None]
        self.r = self.z.clone()
        
    def reset(self):
        """also need to precompute successor matrix"""
        super(DPAgent, self).reset()
        self.values = self.env.vs
        self.update_z_and_r()
        
        return
    
    def step(self, observation):
        """
        Perform one 'update step'
        This involves both choosing an action and updating our value function

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """

        # first update representations
        self.update_z_and_r()
        
        # then compute policy
        next_values = self.values[:, self.t+1, :]
        
        self.pi = (self.beta * next_values).exp() # allocentric policy is just value
        self.pi /= self.pi.sum(-1, keepdims = True) # normalize
        
        if self.env.output_format == "egocentric":
            self.pi = self.allo_to_ego_pi(self.pi) # convert to egocentric

        # and sample an action
        self.action = self.sample_action() # sample an action from the policy
        
        return self.action

