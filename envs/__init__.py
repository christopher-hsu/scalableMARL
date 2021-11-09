""" envs/ folder is for openAIgym-like simulation environments
To use,
>>> import envs
>>> env = envs.make("NAME_OF_ENV")

"""
import gym

def make(env_name, type, render=False, record=False, directory='', **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """

    if type == 'ma_target_tracking':
        import envs.maTTenv
        env = maTTenv.make(env_name, render=render, record=record,
                                                directory=directory, **kwargs)

    else:
        raise ValueError('Designate the right type of the environment.')

    return env
