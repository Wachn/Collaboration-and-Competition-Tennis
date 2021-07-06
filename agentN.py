import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import random
import numpy as np
import copy

from models import Actor, Critic

class DDPG_agent():
    """
    This is based on DDPG algorithm agent
    """
    def __init__(self, state_size, action_size, n_agent, linear_net_dim, seed, batch_size, tau, decay_noise,
                 lr_actor, lr_critic, weight_decay, device):
        """

        :param state_size:
        :param action_size:
        :param linear_net_dim:
        :param seed:
        :param batch_size:
        :param tau:
        :param decay_noise:
        :param lr_actor:
        :param lr_critic:
        :param weight_decay:
        :param device:
        """

        # Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.tau = tau
        self.decay_noise = decay_noise
        self.batch_size = batch_size
        self.device = device

        # Actor
        self.actor_local = Actor(linear_net_dim[0], action_size, state_size, seed).to(device)
        self.actor_target = Actor(linear_net_dim[0], action_size, state_size, seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr =lr_actor, weight_decay=weight_decay)


        # Critic
        self.critic_local = Critic(linear_net_dim[1], action_size, state_size, n_agent, seed).to(device)
        self.critic_target = Critic(linear_net_dim[1], action_size, state_size, n_agent, seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.optimizer_critic  = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)






