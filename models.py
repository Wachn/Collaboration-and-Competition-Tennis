import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


def swish(x):
    return x * torch.sigmoid(x)

def uniform_init(layer):
    nn.init.uniform_(layer.weight.data)
    nn.init.constant_(layer.bias.data, 0)
    return layer

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
        self.actor = nn.ModuleList(
            [uniform_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(self.ddpgActor_dim[:-1], self.ddpgActor_dim[1:])]
        )
        self.actor_feature_dim = self.ddpgActor_dim[-1]
        self.actor_fc = uniform_init(nn.Linear(self.actor_feature_dim, self.action_size))

        # Critic
        self.critic = nn.ModuleList(
            [uniform_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(ddpgCritic_dim[:-1], ddpgCritic_dim[1:])]
        )
        self.critic_feature_dim = self.ddpgCritic_dim[-1]
        self.critic_fc = uniform_init(nn.Linear(self.critic_feature_dim,1))


        # Parameters for backpropagation
        self.actor_params = list(self.actor.parameters()) + list(self.actor_fc.parameters())
        self.critic_params = list(self.critic.parameters()) + list (self.critic_fc.parameters())

        self.optim_actor = optim.Adam(self.actor_params, lr=lr_actor, weight_decay=weight_decay)
        self.optim_critic = optim.Adam(self.critic_params, lr=lr_critic, weight_decay=weight_decay)

    def actor_forward(self, x):
        """Forward action here"""
        for layer in self.actor:
            x = swish(layer(x))
        action = torch.tanh(self)
        return action

    def critic_forward(self, obs_act):
        """Forward critic networ"""
        for layer in self.critic:
            obs_act = swish(obs_act)
        V = self.critic_fc(obs_act)
        return V