import os, copy, pdb
import numpy as np
from numpy import linalg as LA
import gym
from gym import spaces, logger
from gym.utils import seeding
from envs.maTTenv.maps import map_utils
import envs.maTTenv.util as util 
from envs.maTTenv.metadata import METADATA


class maTrackingBase(gym.Env):
    def __init__(self, num_agents=2, num_targets=1, map_name='empty',
                        is_training=True, known_noise=True, **kwargs):
        self.seed()   #used with gym
        self.id = 'maTracking-base'
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * \
                                                len(METADATA['action_w']))
        self.action_map = {}
        for (i,v) in enumerate(METADATA['action_v']):
            for (j,w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v,w)
        assert(len(self.action_map.keys())==self.action_space.n)

        self.agent_dim = 3
        self.target_dim = 2
        self.num_agents = num_agents
        self.nb_agents = num_agents
        self.num_targets = num_targets
        self.nb_targets = num_targets
        self.viewer = None
        self.is_training = is_training

        self.sampling_period = 0.5 # sec
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(
            map_path=os.path.join(map_dir_path, map_name), 
            r_max = self.sensor_r, fov = self.fov/180.0*np.pi,
            margin2wall = METADATA['margin2wall'])

        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = METADATA['target_init_cov']

        self.reset_num = 0
        #needed for gym/core.py wrappers
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

    def seed(self, seed=None):
        '''EXTREMELY IMPORTANT for reproducability in env randomness
        RNG is set and every call of the function will produce the same results
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    def setup_targets(self):
        """Construct all the targets for the environment"""
        raise NotImplementedError

    def setup_belief_targets(self):
        """Construct all the target beliefs for the environment"""
        raise NotImplementedError

    def reset(self, init_random=True):
        """Reset the state of the environment."""
        raise NotImplementedError

    def step(self, action_dict):
        """Takes in dict of action and coverts them to map updates (obs, rewards)"""
        raise NotImplementedError

    def observation(self, target, agent):
        r, alpha = util.relative_distance_polar(target.state[:2],
                                            xy_base=agent.state[:2], 
                                            theta_base=agent.state[2])    
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(map_utils.is_blocked(self.MAP, agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            # z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
            z += self.np_random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        '''For some reason, self.np_random is needed only here instead of np.random in order for the 
        RNG seed to work, if used in the gen_rand_pose functions RNG seed will NOT work '''
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0], #z[0]/self.sensor_r * self.sensor_r_sd, 0.0], 
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def get_reward(self, obstacles_pt, observed, is_training=True):
        return reward_fun(self.belief_targets, is_training)

    def gen_rand_pose(self, o_xy, c_theta, min_lin_dist, max_lin_dist, min_ang_dist, max_ang_dist):
        """Generates random position and yaw.
        Parameters
        --------
        o_xy : xy position of a point in the global frame which we compute a distance from.
        c_theta : angular position of a point in the global frame which we compute an angular distance from.
        min_lin_dist : the minimum linear distance from o_xy to a sample point.
        max_lin_dist : the maximum linear distance from o_xy to a sample point.
        min_ang_dist : the minimum angular distance (counter clockwise direction) from c_theta to a sample point.
        max_ang_dist : the maximum angular distance (counter clockwise direction) from c_theta to a sample point.
        """
        if max_ang_dist < min_ang_dist:
            max_ang_dist += 2*np.pi
        rand_ang = util.wrap_around(np.random.rand() * \
                        (max_ang_dist - min_ang_dist) + min_ang_dist + c_theta)

        rand_r = np.random.rand() * (max_lin_dist - min_lin_dist) + min_lin_dist
        rand_xy = np.array([rand_r*np.cos(rand_ang), rand_r*np.sin(rand_ang)]) + o_xy
        is_valid = not(map_utils.is_collision(self.MAP, rand_xy))
        return is_valid, [rand_xy[0], rand_xy[1], rand_ang]

    def get_init_pose(self, init_pose_list=[], **kwargs):
        """Generates initial positions for the agent, targets, and target beliefs.
        Parameters
        ---------
        init_pose_list : a list of dictionaries with pre-defined initial positions.
        lin_dist_range_target : a tuple of the minimum and maximum distance of a 
                            target from the agent.
        ang_dist_range_target : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a target from the
                            agent. -pi <= x <= pi
        lin_dist_range_belief : a tuple of the minimum and maximum distance of a 
                            belief from the target.
        ang_dist_range_belief : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a belief from the
                            target. -pi <= x <= pi
        blocked : True if there is an obstacle between a target and the agent.
        """
        if init_pose_list:
            self.reset_num += 1
            return init_pose_list[self.reset_num-1]
        else:
            return self.get_init_pose_random(**kwargs)

    def get_init_pose_random(self,
                            lin_dist_range_target=(METADATA['init_distance_min'], METADATA['init_distance_max']),
                            ang_dist_range_target=(-np.pi, np.pi),
                            lin_dist_range_belief=(METADATA['init_belief_distance_min'], METADATA['init_belief_distance_max']),
                            ang_dist_range_belief=(-np.pi, np.pi),
                            blocked=False,
                            **kwargs):
        is_agent_valid = False
        init_pose = {}
        init_pose['agents'] = []
        for ii in range(self.nb_agents):
            is_agent_valid = False
            if self.MAP.map is None and ii==0:
                if blocked:
                    raise ValueError('Unable to find a blocked initial condition. There is no obstacle in this map.')
                a_init = self.agent_init_pos[:2]
            else:
                while(not is_agent_valid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    is_agent_valid = not(map_utils.is_collision(self.MAP, a_init))
            init_pose_agent = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
            init_pose['agents'].append(init_pose_agent)

        init_pose['targets'], init_pose['belief_targets'] = [], []
        for jj in range(self.nb_targets):
            is_target_valid = False
            while(not is_target_valid):
                rand_agent = np.random.randint(self.nb_agents)
                is_target_valid, init_pose_target = self.gen_rand_pose(
                    init_pose['agents'][rand_agent][:2], init_pose['agents'][rand_agent][2],
                    lin_dist_range_target[0], lin_dist_range_target[1],
                    ang_dist_range_target[0], ang_dist_range_target[1])
            init_pose['targets'].append(init_pose_target)

            is_belief_valid, init_pose_belief = False, np.zeros((2,))
            while((not is_belief_valid) and is_target_valid):
                is_belief_valid, init_pose_belief = self.gen_rand_pose(
                    init_pose['targets'][jj][:2], init_pose['targets'][jj][2],
                    lin_dist_range_belief[0], lin_dist_range_belief[1],
                    ang_dist_range_belief[0], ang_dist_range_belief[1])
            init_pose['belief_targets'].append(init_pose_belief)
        return init_pose

def reward_fun(belief_targets, is_training=True, c_mean=0.1):

    detcov = [LA.det(b_target.cov) for b_target in belief_targets]
    r_detcov_mean = - np.mean(np.log(detcov))
    reward = c_mean * r_detcov_mean

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov