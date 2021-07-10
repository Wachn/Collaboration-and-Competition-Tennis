from agentN import DDPG_agent
import torch
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np
import random
import copy


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


def transpose2tensor(x):
    """
    Transpose the N x N-agents x dim ->>> N_agents x N x dim into the form of torch.tensor
    :param x: sample of experiences
    :return: transposed tensor
    """
    tensor_transform = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(tensor_transform, zip(*x)))


class MADDPG:
    def __init__(self, state_size, action_size,n_agent, linear_net_dim, seed, batch_size, buffer_size, tau, gamma, decay_noise,
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
        self.maddpg_agent = [DDPG_agent(state_size=state_size, action_size=action_size, n_agent=n_agent,
                                        linear_net_dim=linear_net_dim, seed=seed, batch_size=batch_size,
                                        tau=tau, decay_noise=decay_noise, lr_actor=lr_actor, lr_critic=lr_critic,
                                        weight_decay=weight_decay, device=device),
                             DDPG_agent(state_size=state_size, action_size=action_size, n_agent=n_agent,
                                        linear_net_dim=linear_net_dim, seed=seed, batch_size=batch_size, tau=tau,
                                        decay_noise=decay_noise, lr_actor=lr_actor, lr_critic=lr_critic,
                                        weight_decay=weight_decay, device=device)]
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.iter = 0
        self.device = device
        self.decay_noise = decay_noise
        self.clip_gradient = False
        self.cl =0
        self.pl =0
        
        # Add Noise
        self.noise = OUNoise(action_size, seed)
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size=buffer_size, seed=seed, device=device)

    def act(self, states, noise=1.0):
        """
        Get the actions from all agents in the MADDPG object
        :param states: Matrix of the observables in numpy,shape of [n_agents, dim]  = (2,24)
        :param noise: Default will be 1.0 else this acts as a switch 0.0 or 1.0
        :return: actions are returned as a list of [agents, (N, dim)]
        """
        actions = []
        states = np.expand_dims(states, axis=1)
        for agent, obs in zip(self.maddpg_agent, states):
            agent.actor_local.eval()
            with torch.no_grad():
                action = agent.actor_local(torch.from_numpy(obs).float().to(self.device)).cpu().detach().numpy()
            agent.actor_local.train()

            # Include noise to this
            self.noise.level()
            action = np.clip(action + np.maximum(self.decay_noise * self.noise.state * noise, 0.001), -1, 1)

            # Update decaying factor for noise
            self.decay_noise = self.decay_noise * np.random.choice([0.999, 1], p=(0.1, 0.9))
            actions.append(action)
        return np.vstack(actions)

    def target_act(self, states):
        """
         Target network actions from all of the agent in MADDPG agent
        :param states : (float) Matrix of the observables in tensor,shape of [n_agent, (batch_size,24)]
        :return: actions are returned as a list of [agents, (N, dim)]
        """
        target_actions = [agent.actor_target(obs) for agent, obs in zip(self.maddpg_agent, states)]
        return target_actions

    def step(self, batch_size, agent_number, logger):
        """
        Learning step to update all the actors and critics of the agent, each agent at a time.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        :param experiences:  List of samples from the replay buffer ( All in tensor send to device already)
        :param agent_number:
        :param logger:
        :return:
        """
        experiences = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = experiences

        agent = self.maddpg_agent[agent_number]
        agent.optimizer_critic.zero_grad()

        # ----------------------------Critic--------------------------#
        target_actions = torch.cat(self.target_act(next_states), dim=1)
     
        with torch.no_grad():
            q_next = agent.critic_target(next_states[agent_number], target_actions)
        q_next = rewards[agent_number].view(-1, 1) + \
                 self.gamma * q_next * (1 - dones[agent_number].view(-1, 1))
        actions = torch.cat(actions, dim=1)
        q = agent.critic_local(states[agent_number], actions)
        critic_loss = F.mse_loss(q, q_next)

        critic_loss.backward()
        
        if self.clip_gradient:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.optimizer_critic.step()

        # ----------------------Actor------------------------------#
        agent.optimizer_actor.zero_grad()

        actions_local = [self.maddpg_agent[i].actor_local(state) if i == agent_number \
                             else self.maddpg_agent[i].actor_local(state).detach() \
                         for i, state in enumerate(states)]
        actions_local = torch.cat(actions_local, dim=1)
        policy_loss = -agent.critic_local(states[agent_number], actions_local).mean()
        policy_loss.backward()

        agent.optimizer_actor.step()
        
        pl = policy_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        assert (pl < 2.0) and (cl<1.0), "Policy Loss blowing Up, {} and {}".format(pl,cl)
        logger.add_scalars("agent%i/losses" % agent_number,
                           {'Critic Loss': cl,
                            'Policy Loss': pl},
                           self.iter)
        self.cl = cl
        self.pl=pl

    def update_targets(self):
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.tau)
            self.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.tau)

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
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
    
    def lr_update(self):
        self.tau*=0.995
        #for ddpg_agent in self.maddpg_agent:
            #ddpg_agent.scheduler_actor.step();
            #ddpg_agent.scheduler_critic.step();


class ReplayBuffer:
    """
    Buffer containing fixed length tuple
    """

    def __init__(self, buffer_size, seed, device):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        self.device = device

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add new experience to the memory buffer
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch of experience
        """
        experiences = random.sample(self.memory, k=batch_size)
        # Convert to torch tensors; first to numpy [n-agents, batch_size, features]
        states = [torch.from_numpy(state).float().to(self.device) for state in
                  np.stack([e.state for e in experiences if e is not None]).transpose((1, 0, 2))]

        actions = [torch.from_numpy(action).float().to(self.device) for action in
                   np.stack([e.action for e in experiences if e is not None]).transpose((1, 0, 2))]

        rewards = [torch.from_numpy(reward).float().to(self.device) for reward in
                   np.stack([e.reward for e in experiences if e is not None]).transpose((1, 0))]

        next_states = [torch.from_numpy(next_state).float().to(self.device) for next_state in
                       np.stack([e.next_state for e in experiences if e is not None]).transpose((1, 0, 2))]

        dones = [torch.from_numpy(done).float().to(self.device) for done in
                 np.stack([e.done for e in experiences if e is not None]).transpose((1, 0))]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
