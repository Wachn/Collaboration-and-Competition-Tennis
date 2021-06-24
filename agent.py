import torch
import torch.nn as nn
import torch.functional as F
import random
import numpy as np
import copy

from models import *


class DDPG_agent():
    """This class object design  the ddpg agent
    """

    def __init__(self, state_size, action_size, ddpg_body_dim, seed, batch_size, tau, decay_noise,
                 lr_actor, lr_critic, weight_decay, device):
        """INitialise an Agent
        :param
            state_siize (int): Dimensions of each states; (n_agents, dims) - (2,24)
            action_size (int): Dimensions of each action; (n_agents, act_dim) - (2,2)
            seed (int): random seed
            batch_size (int): size of each samples
            tau (float): Learning rate of the
            decay_noise (float): OUIS noise will decay with each time step; set to 0 to switch off noise

        """
        # Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.tau = tau
        self.decay_noise = decay_noise
        self.batch_size = batch_size
        self.device = device

        # Actor and critic network
        self.network_local = DDPGmodel(state_size=state_size, action_size=action_size,
                                       ddpgActor_body_dim=ddpg_body_dim[0], ddpgCritic_body_dim=ddpg_body_dim[1],
                                       seed=seed, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay).to(device)
        self.network_target = DDPGmodel(state_size=state_size, action_size=action_size,
                                        ddpgActor_body_dim=ddpg_body_dim[0], ddpgCritic_body_dim=ddpg_body_dim[1],
                                        seed=seed, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay).to(device)
        self.network_target.load_state_dict(self.network_local.state_dict())

        self.noise = OUNoise(action_size, seed)

    def act(self, state, noise=1.0):
        state = state.to(self.device)
        self.noise.level()
        action = self.network_local.actor_forward(state) + torch.from_numpy(self.decay_noise * self.noise.state * noise).to(self.device)

        # Update decaying factor for noise
        self.decay_noise = self.decay_noise * np.random.choice([0.999, 1], p=(0.1, 0.9))

        return action.clip(-1,1)

    def target_act(self, state, noise):
        state = state.to(self.device)
        action = self.network_target.actor_forward(state) + torch.from_numpy(self.decay_noise * self.noise.state * noise).to(self.device)

        return action.clip(-1,1)


class OUNoise:
    """Ornstein_Uhlenbeck Process"""

    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state to the mean (mu)"""
        self.state = copy.copy(self.mu)

    def level(self):
        """Define the update for noise level"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
