import numpy as np
import torch
import torch.nn as nn
from algos.maTT.modules import *

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu', 'SpinningUp']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class SoftActionSelector(nn.Module):
    '''
    Soft parameterization of q value logits,
    pi_log = (1/Z)*(e^(v(x)) - min(v(x))
    If determinstic take max value as action,
    Else (stochastic),
    Sample from multinomial of the soft logits.
    '''
    def __init__(self, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, q, deterministic=False, with_logprob=True):
        q_soft = q - torch.min(q)

        # Convert q values to log probability space
        try:
            pi_log = self.logsoftmax(q_soft)
        except:
            q_soft = q_soft.unsqueeze(0)
            pi_log = self.logsoftmax(q_soft)

        # Select action
        if deterministic:
            mu = torch.argmax(pi_log)
            pi_action = mu      
        else:
            q_log_dist = torch.distributions.multinomial.Multinomial(1, logits=pi_log)
            action = q_log_dist.sample()
            pi_action = torch.argmax(action, dim=1, keepdim=True)

        # Calculate log probability if training
        if with_logprob:
            logp_pi = torch.gather(pi_log,1,pi_action)
        else:
            logp_pi = None
        
        return pi_action, logp_pi

class DeepSetAttention(nn.Module):
    """ Written by Christopher Hsu:

    """
    def __init__(self, dim_input, dim_output, num_outputs=1,
                        dim_hidden=128, num_heads=4, ln=True):
        super().__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    # v(x)
    def values(self, obs):
        v = self.enc(obs)
        v = v.sum(dim=1, keepdim=True)  #pooling mechanism: sum, mean, max
        v = self.dec(v).squeeze()
        return v

    # q(x,a)
    def forward(self, obs, act):
        v = self.enc(obs)
        v = v.sum(dim=1, keepdim=True)  #pooling mechanism: sum, mean, max
        v = self.dec(v).squeeze()
        q = torch.gather(v, 1, act)
        return q

class DeepSetmodel(nn.Module):

    def __init__(self, observation_space, action_space, dim_hidden=128):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = SoftActionSelector(act_dim)
        self.q1 = DeepSetAttention(dim_input=obs_dim, dim_output=act_dim, dim_hidden=dim_hidden)
        self.q2 = DeepSetAttention(dim_input=obs_dim, dim_output=act_dim, dim_hidden=dim_hidden)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            v1 = self.q1.values(obs)
            v2 = self.q2.values(obs)

            a, _ = self.pi(v1+v2, deterministic, False)
            # Tensor to int
            return int(a)
