import torch
import numpy as np
import algos.maTT.core as core

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu', 'SpinningUp']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'

class ReplayBufferSet(object):
    def __init__(self, size, obs_dim, act_dim):
        """Create Replay buffer. edited to include nb_targets_idx buffer

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = {}
        self._maxsize = size
        self._next_idx = 0
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def __len__(self):
        return len(self._storage)

    def store(self, obs_t, action, reward, obs_tp1, done):#, nb_targets):
        nb_targets = obs_t.shape[0]
        try:
            len(self._storage[nb_targets])
        except:
            self._storage[nb_targets] = []
        
        ## Combine into a tuple and append to storage
        data = (obs_t, obs_tp1, action, reward, done)
        
        if self._next_idx >= len(self._storage[nb_targets]):
            self._storage[nb_targets].append(data)
        else:
            self._storage[nb_targets][self._next_idx] = data

        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def _encode_sample(self, size, idxes, nb_targets):
        """Given indexes, return a batch of data
        action data is outputted as an column vector
        """
        batch = dict(obs=np.zeros((size, nb_targets, self._obs_dim)), 
                     obs2=np.zeros((size, nb_targets, self._obs_dim)), 
                     act=np.zeros((size, 1)), 
                     rew=np.zeros((size)), 
                     done=np.zeros((size)))

        for i, idx in enumerate(idxes):
            data = self._storage[nb_targets][idx]
            obs_t, obs_tp1, action, reward, done = data

            batch['obs'][i] = obs_t
            batch['obs2'][i] = obs_tp1
            batch['act'][i] = action
            batch['rew'][i] = reward
            batch['done'][i] = done

        batch['obs'] = torch.as_tensor(batch['obs'], dtype=torch.float32)
        batch['obs2'] = torch.as_tensor(batch['obs2'], dtype=torch.float32)
        batch['act'] = torch.as_tensor(batch['act'])
        batch['rew'] = torch.as_tensor(batch['rew'], dtype=torch.float32)
        batch['done'] = torch.as_tensor(batch['done'])
        return batch

    def sample_batch(self, batch_size):#, num_targets):
        """Sample a batch of experiences. Randomly sample a batch of experiences all
        with the same number of targets. Number of targets is sampled from the range [1,num_targets]

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array [batch, nb_targets, obs_dim]
            batch of observations
        act_batch: np.array [batch, 1]
            batch of actions executed given obs_batch
        rew_batch: np.array [batch]
            rewards received as results of executing act_batch
        next_obs_batch: np.array [batch, nb_targets, obs_dim]
            next set of observations seen after executing act_batch
        done_mask: np.array [batch]
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """

        # nb_targets = np.random.random_integers(1, num_targets)
        nb_targets = np.random.randint(1, len(self._storage))
        idxes = [np.random.randint(0, len(self._storage[nb_targets]) - 1) for _ in range(batch_size)]
        return self._encode_sample(batch_size, idxes, nb_targets)