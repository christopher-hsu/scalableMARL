import pdb, argparse, os, datetime, json, pickle
import torch
import torch.nn as nn

import gym
from gym import wrappers

from algos.maTT.dql import doubleQlearning
import algos.maTT.core as core

import envs

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-2]))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='setTracking-v0')
parser.add_argument('--map', type=str, default="emptyMed")
parser.add_argument('--nb_agents', type=int, default=4)
parser.add_argument('--nb_targets', type=int, default=4)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--mode', choices=['train', 'test', 'test-behavior'], default='train')
parser.add_argument('--steps_per_epoch', type=int, default=25000)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--polyak', type=float, default=0.999) #tau in polyak averaging
parser.add_argument('--hiddens', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_period', type=float, default=0.7) #Back half portion with cosine lr schedule
parser.add_argument('--grad_clip', type=int, default=0.2)
parser.add_argument('--start_steps', type=int, default=20000)
parser.add_argument('--update_after', type=int, default=20000)
parser.add_argument('--num_eval_episodes', type=int, default=2) #During training
parser.add_argument('--replay_size', type=int, default=int(1e6))
parser.add_argument('--max_ep_len', type=int, default=200)
parser.add_argument('--checkpoint_freq', type=int, default=1)

parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--nb_test_eps',type=int, default=50)
parser.add_argument('--log_dir', type=str, default='./results/maTT')
parser.add_argument('--log_fname', type=str, default='model.pt')
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--eval_type', choices=['random', 'fixed_4', 
                                            'fixed_2', 'fixed_nb'], default='fixed_nb')

parser.add_argument('--torch_threads', type=int, default=1)
parser.add_argument('--amp', type=int, default=0)

args = parser.parse_args()


def train(seed, save_dir):
    save_dir_0 = os.path.join(save_dir, 'seed_%d'%seed)

    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=save_dir_0,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=True,
                    )

    # Create env function
    env_fn = lambda : env
    
    #Training function
    model_kwargs = dict(dim_hidden=args.hiddens)
    logger_kwargs = dict(output_dir=save_dir_0, exp_name=save_dir_0)
    model = core.DeepSetmodel

    doubleQlearning(
        env_fn=env_fn,
        model=model,
        model_kwargs=model_kwargs,
        seed=seed, 
        steps_per_epoch=args.steps_per_epoch, 
        epochs=args.epochs, 
        gamma=args.gamma,
        polyak=args.polyak,
        lr=args.learning_rate,
        lr_period=args.learning_rate_period,
        alpha=args.alpha, 
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        start_steps=args.start_steps, 
        update_after=args.update_after,
        num_test_episodes=args.num_eval_episodes,
        replay_size=args.replay_size,
        max_ep_len=args.max_ep_len,
        logger_kwargs=logger_kwargs, 
        save_freq=args.checkpoint_freq, 
        render=bool(args.render),
        torch_threads=args.torch_threads,
        amp=bool(args.amp)
        )

def test(seed):
    from algos.maTT.evaluation import Test, load_pytorch_policy
    
    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )    

    # Load saved policy
    model_kwargs = dict(dim_hidden=args.hiddens)
    model = core.DeepSetmodel(env.observation_space, env.action_space, **model_kwargs)
    policy = load_pytorch_policy(args.log_dir, args.log_fname, model)

    # Testing environment
    Eval = Test()
    Eval.test(args, env, policy)

def testbehavior(seed):
    from algos.maTT.evaluation_behavior import TestBehavior, load_pytorch_policy
    import algos.maTT.core_behavior as core_behavior
    
    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )    

    # Load saved policy
    model_kwargs = dict(dim_hidden=args.hiddens)
    model = core_behavior.DeepSetmodel(env.observation_space, env.action_space, **model_kwargs)
    policy = load_pytorch_policy(args.log_dir, args.log_fname, model)

    # Testing environment
    Eval = TestBehavior()
    Eval.test(args, env, policy)


if __name__ == '__main__':
    if args.mode == 'train':
        save_dir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            ValueError("The directory already exists...", save_dir)

        notes = input("Any notes for this experiment? : ")
        f = open(os.path.join(save_dir, "notes.txt"), 'w')
        f.write(notes)
        f.close()

        seed = args.seed
        list_records = []
        for _ in range(args.repeat):
            print("===== TRAIN A TARGET TRACKING RL AGENT : SEED %d ====="%seed)
            results = train(seed, save_dir)
            json.dump(vars(args), open(os.path.join(save_dir, "seed_%d"%seed, 'learning_prop.json'), 'w'))
            seed += 1
            args.seed += 1

    elif args.mode =='test':
        test(args.seed)

    elif args.mode =='test-behavior':
        testbehavior(args.seed)
