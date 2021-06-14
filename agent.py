import torch
import torch.nn as nn
import torch.functional as F

from models import *

class DDPG_agent():
    """This class object design  the ddpg agent
    """
    def __init__(self, state_size, action_size, ddpg_body_dim, seed, batch_Sizse, tau, decay_noise,
                 lr_actor, lr_critic, weight_decay):
        """INitialise an Agent
        :param
            state_siize (int): Dimensions of each states; (n_agents, dims) - (2,24)
            action_size (int): Dimensions of each action; (n_agents, act_dim) - (2,2)
            seed (int): random seed
            batch_size (int): size of each samples
            tau (float): Learning rate of the
            decay_noise (float): OUIS noise will decay with each time step

        """

        # Actor and critic network
        self.network_local = DDPGmodel(state_size=state_size, action_size=action_size,
                                       ddpgActor_body_dim=ddpg_body_dim[0], ddpgCritic_body_dim=ddpg_body_dim[1],
                                       seed=seed, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=, weight_decay)
        self.network_target = DDPGmodel(state_size=state_size, action_size=action_size,
                                        ddpgActor_body_dim=ddpg_body_dim[0], ddpgCritic_body_dim=ddpg_body_dim[1],
                                        seed=seed, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay)
        self.network_target.load_state_dict(self.network_local.state_dict())

        self.noise = OUNoise(action_size, seed)

    def step(self):



class ReplayBuffer:
    """
    Buffer containing fixed length tuple
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=['state','action','reward','next_state','done'])
        self.seed = random.seed(seed)
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, done ):
        """
        Add new experience to the memory buffer
        """
        exp = self.experience(state,action,reward,next_state,done)
        self.memory.append(exp)

    def sample(self):
        """
        Randomly sample a mini-batch of experience
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states,actions,rewards,next_states,dones)

    def __len__(self):
        return len(self.memory)