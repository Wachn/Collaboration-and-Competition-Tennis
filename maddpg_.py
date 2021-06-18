from agent import DDPG_agent
import torch
from collections import namedtuple, deque
import numpy as np
import random


#device = 'cpu'


def transpose2tensor(x):
    """
    Transpose the N x N-agents x dim ->>> N_agents x N x dim into the form of torch.tensor
    :param x: sample of experiences
    :return: transposed tensor
    """
    tensor_transform = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(tensor_transform, zip(*x)))


# ddpg_body_dim = [(128,256), (128,256)]
class MADDPG:
    def __init__(self, state_size, action_size, ddpg_body_dim, seed, batch_size, buffer_size, tau, gamma, decay_noise,
                 lr_actor, lr_critic, weight_decay, device):
        """
        Create a multiagent that consist of of multiple ddpg agents
        Params
        =======
        :param state_size: Number of observables - 24
        :param action_size: Possible action moves by agent - 2
        :param ddpg_body_dim: E.g., [(256,128), (256,128)]
        :param seed: 0
        :param batch_size: 128
        :param tau: Soft-update learning rate
        :param gamma: Define the discount rate
        :param decay_noise: Set 0 to tune off noise input, else <1
        :param lr_actor: Learning rate of local_actor
        :param lr_critic: Learning rate of local_critic
        :param weight_decay: weight decay of both actor and critic
        """

        # Critic input = state_size + actions = 24 + 2 + 2 = 28
        self.maddpg_agent = [DDPG_agent(state_size=state_size, action_size=action_size, ddpg_body_dim=ddpg_body_dim,
                                        seed=seed, batch_size=batch_size, tau=tau, decay_noise=decay_noise,
                                        lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device),
                             DDPG_agent(state_size=state_size, action_size=action_size, ddpg_body_dim=ddpg_body_dim,
                                        seed=seed, batch_size=batch_size, tau=tau, decay_noise=decay_noise,
                                        lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device)
                             ]
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.iter = 0

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed)

    def acts(self, states_all_agents, noise=1.0):
        """
        Get the actions from all agents in the MADDPG object
        :param states_all_agents (float): Matrix of the observables in numpy,shape of [n_agents, dim]  = (2,24)
        :param noise: Default will be 1.0 else this acts as a switch 0.0 or 1.0
        :return: actions are returned as a list of [agents, (N, dim)]
        """
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, states_all_agents)]
        return actions

    def target_acts(self, states_all_agents, noise=1.0):
        """
        Target network actions from all of the agent in MADDPG agent
        :param states_all_agents (float): Matrix of the observables in numpy,shape of [n_agents, dim]  = (2,24)
        :param noise: Default will be 1.0 else this acts as a switch 0.0 or 1.0
        :return: actions are returned as a list of [agents, (N, dim)]
        """
        target_actions = [agent.target_act(obs, noise) for agent, obs in zip(self.maddpg_agent, states_all_agents)]
        return target_actions

    def step(self, experiences, agent_number, gamma, logger):
        """
        Learning step to update all the actors and critics of the agent, each agent at a time.
         Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        :param experiences:  List of samples from the replay buffer
        :param agent_number:
        :param gamma:
        :param logger:
        :return:
        """
        # Transform to tensor: [n_agent x N x dim]
        states, actions, rewards, next_states, dones = map(transpose2tensor, experiences)

        agent = self.maddpg_agent[agent_number]
        agent.network_local.optim_critic.zero_grad()

        # -----------------------Critic----------------------------#
        target_actions = self.target_acts(next_states)
        # Fuse all the actions together to form [N x dim matrix]
        target_actions = torch.cat(target_actions, dim=1)
        critic_target_obs = torch.cat((next_states[agent_number], target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.network_target.critic_forward(critic_target_obs)
        q_next = rewards[agent_number].view(-1, 1) + self.gamma * q_next * (1 - dones[agent_number].view(-1, 1))
        actions = torch.cat(actions, dim=1)
        critic_local_obs = torch.cat((states[agent_number], actions), dim=1).to(device)
        q = agent.network_local.critic_forward(critic_local_obs)

        hubber_loss = torch.nn.SmoothL1Loss()
        critic_loss = hubber_loss(q, q_next.detach())
        critic_loss.backward()

        agent.network_local.optim_critic.step()

        # -------------Actor--------------#
        agent.network_local.optim_actor.zero_grad()
        actor_local_actions = [self.maddpg_agent[i].act(state) if i == agent_number \
                                   else self.maddpg_agent[i].act(state).detach()
                               for i, state in enumerate(states)]
        actor_local_obs = torch.cat((states[agent_number], torch.cat(actor_local_actions, dim=1)), dim=1).to(device)
        policy_loss = -agent.network_local.critic_forward(actor_local_obs)
        policy_loss.backward()

        agent.network_local.optim_actor.step()

        pl = policy_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars("agent%i/losses" % agent_number,
                           {'Critic Loss': cl,
                            'Policy Loss': pl},
                           self.iter)

    def update_targets(self):
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.network_local, ddpg_agent.network_target, self.tau)

    def soft_update(self, local_net, target_net, tau):
        """
        Soft update target model parameters
        Eqn: θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_net (pytorch): weights to be copied
            target_net (pytorch): weights to be updated to
            tau (float): Learning interpolation parameter for target model
        """
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


class ReplayBuffer:
    """
    Buffer containing fixed length tuple
    """

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add new experience to the memory buffer
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """
        Randomly sample a mini-batch of experience
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        # Convert to torch tensors
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
