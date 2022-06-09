import datetime, json, os, argparse, time
import pickle, tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os.path as osp
import torch

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'

def load_pytorch_policy(fpath, fname, model, deterministic=True):
    fname = osp.join(fpath,'state_dict/',fname)
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(fname, map_location))

    # make function for producing an action given a single state
    def get_action(x, deterministic=True):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            action = model.act(x, deterministic)
        return action

    return get_action

def eval_set(num_agents, num_targets):
    agents = np.linspace(num_agents/2, num_agents, num=3, dtype=int)
    targets = np.linspace(num_agents/2, num_targets, num=3, dtype=int)
    params_set = [{'nb_agents':1, 'nb_targets':1},
                  {'nb_agents':4, 'nb_targets':4}]
    for a in agents:
        for t in targets:
            params_set.append({'nb_agents':a, 'nb_targets':t})
    return params_set

class Test:
    def __init__(self):
        pass

    def test(self, args, env, act, torch_threads=1):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_num_threads(torch_threads)

        seed = args.seed
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        set_eval = eval_set(env.num_agents, env.num_targets)
        if args.eval_type == 'random':
            params_set = [{}]
        elif args.eval_type == 'fixed_nb':
            ## Either manually set evaluation set or auto fill
            params_set = SET_EVAL_v0
            # params_set = set_eval
        else:
            raise ValueError("Wrong evaluation type for ttenv.")

        timelimit_env = env
        while( not hasattr(timelimit_env, '_elapsed_steps')):
            timelimit_env = timelimit_env.env

        total_nlogdetcov = []
        total_intruders = []
        for params in params_set:
            ep = 0
            ep_nlogdetcov = [] #'Episode nLogDetCov'
            ep_intruders = []
            time_elapsed = ['Elapsed Time (sec)']

            while(ep < args.nb_test_eps): # test episode
                ep += 1
                s_time = time.time()
                episode_rew, nlogdetcov, ep_len, intruders = 0, 0, 0, 0
                done = {'__all__':False}
                obs = env.reset(**params)

                while not done['__all__']:
                    if args.render:
                        env.render()
                    action_dict = {}
                    for agent_id, o in obs.items():
                        action_dict[agent_id] = act(o, deterministic=False)
                    obs, rew, done, info = env.step(action_dict)
                    episode_rew += rew['__all__']
                    nlogdetcov += info['mean_nlogdetcov']
                    ep_len += 1

                time_elapsed.append(time.time() - s_time)
                ep_nlogdetcov.append(nlogdetcov)

                if args.render:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))
                if ep % 50 == 0:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))

            if args.record :
                env.moviewriter.finish()
            if args.ros_log :
                ros_log.save(args.log_dir)

            # Stats
            meanofeps = np.mean(ep_nlogdetcov)
            total_nlogdetcov.append(meanofeps)
            # Eval plots and saves
            if args.env == 'setTracking-vGreedy':
                eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'greedy_eval_seed%d_'%(seed)+args.map)
            else:
                eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'eval_seed%d_'%(seed)+args.map)
            model_seed = os.path.split(args.log_dir)[-1]           
            # eval_dir = os.path.join(args.log_dir, 'eval_seed%d_'%(seed)+args.map)
            # model_seed = os.path.split(args.log_fname)[0]
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            matplotlib.use('Agg')
            f0, ax0 = plt.subplots()
            _ = ax0.plot(ep_nlogdetcov, '.')
            _ = ax0.set_title(args.env)
            _ = ax0.set_xlabel('episode number')
            _ = ax0.set_ylabel('mean nlogdetcov')
            _ = ax0.axhline(y=meanofeps, color='r', linestyle='-', label='mean over episodes: %.2f'%(meanofeps))
            _ = ax0.legend()
            _ = ax0.grid()
            _ = f0.savefig(os.path.join(eval_dir, "%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, args.nb_test_eps)
                                                    +model_seed+".png"))
            plt.close()
            pickle.dump(ep_nlogdetcov, open(os.path.join(eval_dir,"%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, args.nb_test_eps))
                                                                    +model_seed+".pkl", 'wb'))

        #Plot over all example episode sets
        f1, ax1 = plt.subplots()
        _ = ax1.plot(total_nlogdetcov, '.')
        _ = ax1.set_title(args.env)
        _ = ax1.set_xlabel('example episode set number')
        _ = ax1.set_ylabel('mean nlogdetcov over episodes')
        _ = ax1.grid()
        _ = f1.savefig(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_eps)+model_seed+'.png'))
        plt.close()        
        pickle.dump(total_nlogdetcov, open(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_eps))+model_seed+'%da%dt'%(args.nb_agents,args.nb_targets)+'.pkl', 'wb'))

SET_EVAL_v0 = [
        # {'nb_agents': 1, 'nb_targets': 1},
        # {'nb_agents': 2, 'nb_targets': 1},
        # {'nb_agents': 3, 'nb_targets': 1},
        # {'nb_agents': 4, 'nb_targets': 1},
        # {'nb_agents': 1, 'nb_targets': 2},
        # {'nb_agents': 2, 'nb_targets': 2},
        # {'nb_agents': 3, 'nb_targets': 2},
        # {'nb_agents': 4, 'nb_targets': 2},
        # {'nb_agents': 1, 'nb_targets': 3},
        # {'nb_agents': 2, 'nb_targets': 3},
        # {'nb_agents': 3, 'nb_targets': 3},
        # {'nb_agents': 4, 'nb_targets': 3},
        # {'nb_agents': 1, 'nb_targets': 4},
        # {'nb_agents': 2, 'nb_targets': 4},
        # {'nb_agents': 3, 'nb_targets': 4},
        {'nb_agents': 20, 'nb_targets': 20},
]