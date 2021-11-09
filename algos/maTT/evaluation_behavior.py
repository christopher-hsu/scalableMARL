import datetime, json, os, argparse, time, pdb
import pickle, tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter

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


def get_init_pose_list(nb_test_eps, eval_type):
    init_pose_list = []
    if eval_type == 'fixed_2':
        for ii in range(nb_test_eps):
            left = np.random.uniform(20,30)
            Lyaxis = np.random.uniform(25,35)
            right = np.random.uniform(30,40)
            Ryaxis = np.random.uniform(35,45)
            init_pose_list.append({'agents':[[24.5, 15.5, 1.57], [26.5, 15.5, 1.57]],
                            'targets':[[left, Lyaxis, 0, 0],[right, Ryaxis, 0, 0]],
                            'belief_targets':[[left, Lyaxis, 0, 0], [right, Ryaxis, 0, 0]]})
    else:
        for ii in range(nb_test_eps):
            xone = np.random.uniform(20,30)
            yone = np.random.uniform(15,25)
            xtwo = np.random.uniform(30,40)
            ytwo = np.random.uniform(25,35)
            xthree = np.random.uniform(20,30)
            ythree = np.random.uniform(35,45)
            xfour = np.random.uniform(10,20)
            yfour = np.random.uniform(35,45)
            init_pose_list.append({'agents':[[24.5, 10, 1.57], [26.5, 10, 1.57], 
                                            [22.5, 10, 1.57], [28.5, 10, 1.57]],
                            'targets':[[xone, yone, 0, 0],[xtwo, ytwo, 0, 0],
                                        [xthree, ythree, 0, 0],[xfour, yfour, 0, 0]],
                            'belief_targets':[[xone, yone, 0, 0], [xtwo, ytwo, 0, 0],
                                            [xthree, ythree, 0, 0], [xfour, yfour, 0, 0]]}) 

    return init_pose_list

class TestBehavior:
    def __init__(self):
        pass

    def test(self, args, env, act, torch_threads=1):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_num_threads(torch_threads)

        seed = args.seed
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'behave_seed%d_'%(seed)+args.map)
        model_seed = os.path.split(args.log_dir)[-1]           
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        behave_dir = os.path.join(os.path.split(args.log_dir)[0], 'behave_seed%d_'%(seed)+args.map)
        model_seed = os.path.split(args.log_dir)[-1]           
        if not os.path.exists(behave_dir):
            os.makedirs(behave_dir)
        # writer = SummaryWriter(behave_dir)

        if args.eval_type == 'random':
            params_set = [{}]
        elif args.eval_type == 'fixed_2':
            params_set = EVAL_BEHAVIOR_2
            tot_eplen = 60
        elif args.eval_type == 'fixed_4':
            params_set = EVAL_BEHAVIOR_4
            tot_eplen = 100
        else:
            raise ValueError("Wrong evaluation type for ttenv.")

        timelimit_env = env
        while( not hasattr(timelimit_env, '_elapsed_steps')):
            timelimit_env = timelimit_env.env

        if args.ros_log:
            from envs.target_tracking.ros_wrapper import RosLog
            ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)

        init_pose_list = get_init_pose_list(args.nb_test_eps, args.eval_type)

        total_nlogdetcov = []
        for params in params_set:
            ep = 0
            ep_nlogdetcov = [] #'Episode nLogDetCov'
            time_elapsed = ['Elapsed Time (sec)']
            test_observations = np.zeros(args.nb_test_eps)
            
            while(ep < args.nb_test_eps): # test episode
                ep += 1
                s_time = time.time()
                episode_rew, nlogdetcov, ep_len = 0, 0, 0
                done = {'__all__':False}
                obs = env.reset(init_pose_list=init_pose_list, **params)

                all_observations = np.zeros(env.nb_targets, dtype=bool)


                bigq0 = []
                bigq1 = []

                while ep_len < tot_eplen:
                    if args.render:
                        env.render()
                    if args.ros_log:
                        ros_log.log(env)
                    action_dict = {}
                    q_dict = {}
                    for agent_id, o in obs.items():
                        action_dict[agent_id], q_dict[agent_id] = act(o, deterministic=False)
                        # record target observations
                        observed = np.zeros(env.nb_targets, dtype=bool)
                        all_observations = np.logical_or(all_observations, o[:,5].astype(bool))  

                    if all(all_observations) == True:
                        test_observations[ep-1] = 1

                    obs, rew, done, info = env.step(action_dict)
                    episode_rew += rew['__all__']
                    nlogdetcov += info['mean_nlogdetcov']
                    # log q values
                    rearrange = [0,3,6,9,1,4,7,10,2,5,8,11]
                    q0 = np.zeros((12))
                    q1 = np.zeros((12))
                    for ii, val in enumerate(rearrange):
                        qs0 = q_dict['agent-0'].squeeze(0)
                        qs1 = q_dict['agent-1'].squeeze(0)
                        q0[ii] = qs0[val]
                        q1[ii] = qs1[val] 

                    bigq0.append(q0)
                    bigq1.append(q1)
                    ep_len += 1

                bigq0 = np.asarray(bigq0)
                bigq1 = np.asarray(bigq1)


                time_elapsed.append(time.time() - s_time)
                ep_nlogdetcov.append(nlogdetcov)
                if args.render:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f, ep len : %d"%(ep, episode_rew, nlogdetcov, ep_len))
                if ep % 50 == 0:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))

            if args.record :
                env.moviewriter.finish()
            if args.ros_log :
                ros_log.save(args.log_dir)

            print(test_observations)
            print("Cooperation ratio over total evals: %.2f"%(np.sum(test_observations)/args.nb_test_eps))

            # plot actions for agents in 3d bar graph
            f2 = plt.figure()
            ax2 = f2.add_subplot(121,projection='3d')
            ax3 = f2.add_subplot(122,projection='3d')


            lx = len(bigq0[0])
            ly = len(bigq0[:,0])
            xpos = np.arange(0,lx,1)
            ypos = np.arange(0,ly,1)
            xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros(lx*ly)

            dx = 0.5 *np.ones_like(zpos)
            dy = dx.copy()
            dz0 = bigq0.flatten()
            dz1 = bigq1.flatten()
            
            cs = ['r', 'r', 'r', 'r', 'g', 'g', 'g', 'g','b','b','b','b'] * ly

            ax2.bar3d(xpos,ypos,zpos, dx, dy, dz0, color=cs)
            ax3.bar3d(xpos,ypos,zpos, dx, dy, dz1, color=cs)
            # plt.show()

EVAL_BEHAVIOR_2 = [
        {'nb_agents': 2, 'nb_targets': 2},
]
EVAL_BEHAVIOR_4 = [
        {'nb_agents': 4, 'nb_targets': 4},
]
