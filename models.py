import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np


def swish(x):
    return x * torch.sigmoid(x)

def uniform_init(layer):
    nn.init.uniform_(layer.weight.data, *hidden_init(layer))
    nn.init.constant_(layer.bias.data, 0)
    return layer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt((fan_in))
    return ((-lim,lim))


# placeholder for ddpg dimensions
ddpgActor_body_dim = (256, 128)
ddpgCritic_body_dim = (256, 128)

# States = (n_agents, states_dim)
# actions = (n_agents, actions_len)
class DDPGmodel(nn.Module):
    """Model designed for DDPG"""

    def __init__(self, state_size, action_size, ddpgActor_body_dim, ddpgCritic_body_dim, seed, lr_actor, lr_critic, weight_decay):
        super(DDPGmodel, self).__init__()
        """
        Initialise DDPG model for DDPG algorithm, consist of both actor and crictic method"""
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.ddpgActor_dim = (state_size,) + ddpgActor_body_dim
        self.ddpgCritic_dim = (state_size + 2*action_size,) + ddpgCritic_body_dim

        # Actor
        self.actor = DDPG_actor(self.ddpgActor_dim, action_size)

        # Critic
        self.critic = DDPG_critic(self.ddpgCritic_dim)

        # Parameters for backpropagation
        self.actor_params = list(self.actor.actor.parameters()) + list(self.actor.actor_fc.parameters())
        self.critic_params = list(self.critic.critic.parameters()) + list (self.critic.critic_fc.parameters())

        self.optim_actor = optim.Adam(self.actor_params, lr=lr_actor, weight_decay=weight_decay)
        self.optim_critic = optim.Adam(self.critic_params, lr=lr_critic, weight_decay=weight_decay)

    def actor_forward(self, x):
        """Forward action here"""

        action = self.actor.forward(x)
        return action

    def critic_forward(self, obs_act):
        """Forward critic networ"""
        
        V = self.critic.forward(obs_act)
        return V
    
class DDPG_actor(nn.Module):
    """DDPG Actor"""

    def __init__(self, ddpgActor_dim, action_size):
        super(DDPG_actor, self).__init__()
        self.ddpgActor_dim = ddpgActor_dim
        self.action_size = action_size
        
        self.actor = nn.ModuleList(
            [uniform_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in
             zip(self.ddpgActor_dim[:-1], self.ddpgActor_dim[1:])]
        )
        self.actor_feature_dim = self.ddpgActor_dim[-1]
        self.actor_fc = uniform_init(nn.Linear(self.actor_feature_dim, self.action_size))
        self.actor_bn = nn.BatchNorm1d(self.ddpgActor_dim[0])

    def forward(self, x):
        """Forward action here"""
        x = self.actor_bn(x)
        for layer in self.actor:
            x = swish(layer(x))
        action = torch.tanh(self.actor_fc(x))
        return action


class DDPG_critic(nn.Module):
    """DDPG Critic"""

    def __init__(self, ddpgCritic_dim):
        super(DDPG_critic, self).__init__()
        self.ddpgCritic_dim = ddpgCritic_dim
        self.critic = nn.ModuleList(
            [uniform_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in
             zip(self.ddpgCritic_dim[:-1], self.ddpgCritic_dim[1:])]
        )
        self.critic_feature_dim = self.ddpgCritic_dim[-1]
        self.critic_fc = uniform_init(nn.Linear(self.critic_feature_dim, 1))
        self.critic_bn = nn.BatchNorm1d(self.ddpgCritic_dim[0])

    def forward(self, obs_act):
        """Forward critic networ"""
        obs_act = self.critic_bn(obs_act)
        for layer in self.critic:
            obs_act = swish(layer(obs_act))
        V = self.critic_fc(obs_act)
        return V

class Actor(nn.Module):
    """Generic model  for MADDPG"""
    def __init__(self, linear_dim, action_size, state_size, seed):
        """
        Params
        ======
        :param linear_dim:  (list) e.g.[256,128]
        :param action_size: (int) 2
        :param state_size:  (int 8
        :param seed:  (int) 0
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.net_body_dim = (state_size,) + linear_dim
        self.seed = torch.manual_seed(seed)

        self.net_body = nn.ModuleList(
                        [nn.Linear(dim_in, dim_out) for dim_in, dim_out in
                         zip(self.net_body_dim[:-1], self.net_body_dim[1:])]
        )
        self.fc = nn.Linear(self.net_body_dim[-1], action_size)
        self.reset_parameters()
        self.bn = nn.BatchNorm1d(self.net_body_dim[1])

    def reset_parameters(self):
        for i in range(len(self.net_body)):
            self.net_body[i].weight.data.uniform_(*hidden_init(self.net_body[i]))
        self.fc.weight.data.uniform_(-3e-3,3e-3)

    def forward(self, obs):
        """Actor net map states to actions"""
        for i, layer in enumerate(self.net_body):
            if i==1:
                obs = self.bn(obs)
            obs = swish(layer(obs))
        act = torch.tanh(self.fc(obs))
        return act

class Critic(nn.Module):
    """Generic Critic for MADDPG"""
    def __init__(self, linear_dim, action_size, state_size, n_agent, seed):
        """
        Params
        =======
        :param linear_dim: (list: int) E.g. [300,256]
        :param action_size: (int) 2
        :param state_size: (int) 8
        :param n_agent: (int) 2
        :param seed: (int)
        """
        super(Critic, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)
        self.net_body_dim = ((action_size*n_agent)+state_size,) + linear_dim

        self.net_body = nn.ModuleList(
                        [nn.Linear(in_dim, out_dim) for in_dim, out_dim in
                         zip(self.net_body_dim[:-1], self.net_body_dim[1:])]
                        )
        self.fc = nn.Linear (linear_dim[-1], 1)
        self.reset_parameters()
        self.bn = nn.BatchNorm1d(self.net_body_dim[1])

    def reset_parameters(self):
        for i in range(len(self.net_body)):
            self.net_body[i].weight.data.uniform_(*hidden_init(self.net_body[i]))
        self.fc.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, actions):
        """Actor net map states to actions
        :param actions: (float) (batch_size, n_agents x actions)
        :param obs: (float) (batch_size, state)
        """
        x = torch.cat((obs, actions), dim=1).float()

        for i, layer in enumerate(self.net_body):
            if i==1:
                x = self.bn(x)
            x = swish(layer(x))
        x = self.fc(x)
        return x

