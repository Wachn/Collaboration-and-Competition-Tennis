import torch
import torch.nn as nn
import torch.functional as F

from models import *

class DDPG_agent():
    """This class object design  the ddpg agent
    """
    def __init__(self, state_size, action_size, ddpg_body_dim, seed, batch_Sizse, tau, decay_noise):
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
                                       seed=seed)
        self.network_target = DDPGmodel(state_size=state_size, action_size=action_size,
                                        ddpgActor_body_dim=ddpg_body_dim[0], ddpgCritic_body_dim=ddpg_body_dim[1],
                                        seed=seed)
        self.network_target.load_state_dict(self.network_local.state_dict())


    


