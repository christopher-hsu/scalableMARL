import maTTenv
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='setTracking-v0')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--nb_agents', help='the number of agents', type=int, default=4)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=4)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str, default='.')
parser.add_argument('--map', type=str, default="emptyMed")

args = parser.parse_args()

# @profile
def main():
    env = maTTenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )
    nlogdetcov = []
    action_dict = {}
    done = {'__all__':False}

    obs = env.reset()
    # See below why this check is needed for training or eval loop
    while not done['__all__']:
        if args.render:
            env.render()

        for agent_id, o in obs.items():
            action_dict[agent_id] = env.action_space.sample()

        obs, rew, done, info = env.step(action_dict)
        nlogdetcov.append(info['mean_nlogdetcov'])

    print("Sum of negative logdet of the target belief covariances : %.2f"%np.sum(nlogdetcov))

if __name__ == "__main__":
    main()
    """
    To use line_profiler
    add @profile before a function to profile
    kernprof -l run_ma_example.py --env setTracking-v3 --nb_agents 4 --nb_targets 4 --render 0
    python -m line_profiler run_ma_example.py.lprof 

    Examples:
        >>> env = MyMultiAgentEnv()
        >>> obs = env.reset()
        >>> print(obs)
        {
            "agent_0": [2.4, 1.6],
            "agent_1": [3.4, -3.2],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
                "agent_0": 1, "agent_1": 0,
            })
        >>> print(rew)
        {
            "agent_0": 3,
            "agent_1": -1,
            "__all__": 2,
        }
        >>> print(done)
        #Due to gym wrapper, done at TimeLimit is bool, True.
        #During episode, it is a dict so..
        #While done is a dict keep running
        {
            "agent_0": False,  # agent_0 is still running
            "agent_1": True,   # agent_1 is done
            "__all__": False,  # the env is not done
        }
        >>> print(info)
        {
            "agent_0": {},  # info for agent_0
            "agent_1": {},  # info for agent_1
        }
    """