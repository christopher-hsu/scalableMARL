from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.spaces import Box, Discrete
import time, os, random, pdb
from utils.logSpinUp import EpochLogger
import algos.maTT.core as core
from algos.maTT.replay_buffer import ReplayBufferSet as ReplayBuffer

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu', 'SpinningUp']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'


SET_EVAL_v0 = [
        {'nb_agents': 1, 'nb_targets': 1},
        {'nb_agents': 2, 'nb_targets': 2},
        {'nb_agents': 3, 'nb_targets': 3},
        {'nb_agents': 4, 'nb_targets': 4},
]

def eval_set(num_agents, num_targets):
    agents = np.linspace(num_agents/2, num_agents, num=3, dtype=int)
    targets = np.linspace(num_agents/2, num_targets, num=3, dtype=int)
    params_set = [{'nb_agents':1, 'nb_targets':1},
                  {'nb_agents':4, 'nb_targets':4}]
    for a in agents:
        for t in targets:
            params_set.append({'nb_agents':a, 'nb_targets':t})
    return params_set

def test_agent(test_env, get_action, logger, num_test_episodes, 
                num_agents, num_targets, render=False):
    """ Evaluate current policy over an environment set
    """
    ## Either manually set evaluation set or auto fill
    params_set = SET_EVAL_v0
    # params_set = eval_set(num_agents, num_targets)

    for params in params_set:
        for j in range(num_test_episodes):
            done, ep_ret, ep_len = {'__all__':False}, 0, 0
            obs = test_env.reset(**params)
            while not done['__all__']:
                if render:
                    test_env.render()
                action_dict = {}
                for agent_id, o in obs.items():
                    action_dict[agent_id] = get_action(o, deterministic=False)

                obs, rew, done, _ = test_env.step(action_dict)
                ep_ret += rew['__all__']
                ep_len += 1  
            logger.store(TestEpRet=ep_ret)


def doubleQlearning(env_fn, model=core.DeepSetmodel, model_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-2, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=4000, update_every=1, num_test_episodes=5, max_ep_len=200, 
        logger_kwargs=dict(), save_freq=1, lr_period=0.7, grad_clip=5, render=False,
        torch_threads=1, amp=False):
    """
    Soft Clipped Double Q-learning

    Written by Christopher Hsu
    based on SpinningUp structure

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        policy: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations. When called, ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (act_dim)         | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and action.
            ``q1.values``(batch,act_dim)   | Tensor containing current estimate
                                           | of Q*'s' for observations.
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and action.
            ``q2.values``(batch,act_dim)   | Tensor containing current estimate
                                           | of Q*'s' for observations.
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        model_kwargs (dict): Any kwargs appropriate for the MLPmodel object 
            you provided to double q learning.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(torch_threads)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    writer = SummaryWriter(logger_kwargs['output_dir'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    # env, test_env = env_fn(), env_fn()    #Can be used if env is official gym env
    env = env_fn()
    test_env = deepcopy(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    policy = model(env.observation_space, env.action_space, **model_kwargs).to(device)
    policy_targ = deepcopy(policy).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in policy_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(policy.q1.parameters(), policy.q2.parameters())

    # Experience Replay Buffer
    replay_buffer = ReplayBuffer(replay_size, obs_dim, act_dim)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [policy.q1, policy.q2])
    logger.log('\nNumber of parameters: \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up optimizer
    q_optimizer = Adam(q_params, lr=lr)

    #AMP: automatic mixed precision: fp32 -> fp 16
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    # Set up function for computing Q-losses
    def compute_loss_q(data):
        obs = data['obs'].to(device)
        act = data['act'].type(torch.LongTensor).to(device)
        rew = data['rew'].to(device)
        obs2 = data['obs2'].to(device)
        done = data['done'].type(torch.float32).to(device)  

        q1 = policy.q1(obs,act)
        q2 = policy.q2(obs,act)

        # Bellman backup for Q functions
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                # Target actions come from *current* policy
                v1 = policy.q1.values(obs2)
                v2 = policy.q2.values(obs2)
                act2, logp_a2 = policy.pi(v1+v2)

                # Target Q-values
                q1_pi_targ = policy_targ.q1(obs2, act2)
                q2_pi_targ = policy_targ.q2(obs2, act2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

                #Unsqueeze adds another dim, necessary to be column vectors
                # backup = r.unsqueeze(1) + gamma * (1 - done).unsqueeze(1) * (q_pi_targ - alpha * logp_a2)
                backup = rew.unsqueeze(1) + gamma * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        # loss_q1 = ((q1 - backup)**2).mean()
        # loss_q2 = ((q2 - backup)**2).mean()
        # Huber loss against Bellman backup
        huber = torch.nn.SmoothL1Loss()
        loss_q1 = huber(q1, backup)
        loss_q2 = huber(q2, backup)
        loss_q = loss_q1 + loss_q2

        # Useful info for logging, if training on gpu need to send to cpu to store
        try:
            q_info = dict(Q1Vals=q1.detach().numpy(),
                          Q2Vals=q2.detach().numpy())
        except:
            q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                          Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    def update(data, lr_iter):
        # Update learning rate with cosine schedule
        lr = np.clip(0.0005*np.cos(np.pi*lr_iter/(total_steps*lr_period))+0.000501, 1e-5, 1e-3)
        q_optimizer.param_groups[0]['lr'] = lr

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()

        # Automatic Mixed Precision, used for faster training with fp16 operations
        if amp:
            # Enables autocasting for the forward pass (model + loss)
            with torch.cuda.amp.autocast():
                loss_q, q_info = compute_loss_q(data)
            # Calls backward() on scaled loss to create scaled gradients
            scaler.scale(loss_q).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(q_optimizer)
            # Clip gradient values
            torch.nn.utils.clip_grad_value_(policy.parameters(), grad_clip)
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            scaler.step(q_optimizer)
            # Updates the scale for next iteration.
            scaler.update()

        # Standard fp32 training operations
        else:
            loss_q, q_info = compute_loss_q(data)
            # Backprop
            loss_q.backward()
            # Clip gradient values
            torch.nn.utils.clip_grad_value_(policy.parameters(), grad_clip)
            # Gradient step
            q_optimizer.step()

        ## Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(policy.parameters(), policy_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(obs, deterministic=False):
        # Unsqueeze obs to [1, n, d]
        return policy.act(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device), deterministic)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    ep_ret, ep_len, best_test_ret = 0, 0, 0
    obs = env.reset()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        action_dict = {}
        if t > start_steps:
            for agent_id, o in obs.items():
                action_dict[agent_id] = get_action(o, deterministic=False)
        else:
            for agent_id, o in obs.items():
                action_dict[agent_id] = env.action_space.sample()
        
        # Step the env
        obs2, rew, done, info = env.step(action_dict)
        ep_ret += rew['__all__']
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done['__all__'] = False if ep_len==max_ep_len else False

        # Store experience to replay buffer seperately for each agent
        for agent_id, o in obs.items():
            replay_buffer.store(o, action_dict[agent_id], rew['__all__'], 
                                obs2[agent_id], float(done['__all__']))#, env.nb_targets)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        obs = obs2

        # End of trajectory handling
        if done['__all__'] or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_ret, ep_len = 0, 0
            obs = env.reset()

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                #Cosine learning rate schedule
                if t < total_steps*(1-lr_period):
                    lr_iter = 0
                else:
                    lr_iter = t-total_steps*(1-lr_period)

                batch = replay_buffer.sample_batch(batch_size)#, env.num_targets)
                update(data=batch, lr_iter=lr_iter)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            logger.log_tabular('Epoch', epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent(test_env, get_action, logger, num_test_episodes, env.num_agents, env.num_targets)
            # Averages test ep returns
            logger.log_tabular('TestEpRet', with_min_and_max=True)

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     torch.save(policy.state_dict(), fpath+'model%d.pt'%epoch)

            # Save model based on best test episode return
            if logger.log_current_row['AverageTestEpRet'] > best_test_ret:
                logger.log('Saving model, AverageTestEpRet increase %d -> %d'%
                            (best_test_ret, logger.log_current_row['AverageTestEpRet']))

                fpath = logger_kwargs['output_dir']+'/state_dict/'
                os.makedirs(fpath, exist_ok=True)
                torch.save(policy.state_dict(), fpath+'model.pt')
                best_test_ret = logger.log_current_row['AverageTestEpRet']


            # Log info about epoch
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LR', q_optimizer.param_groups[0]['lr'])
            logger.log_tabular('Time', time.time()-start_time)

            # Tensorboard logger
            writer.add_scalar('AverageEpRet', logger.log_current_row['AverageEpRet'],t)
            writer.add_scalar('AverageTestEpRet', logger.log_current_row['AverageTestEpRet'],t)
            writer.add_scalar('AverageQ1Vals', logger.log_current_row['AverageQ1Vals'],t)
            writer.add_scalar('AverageQ2Vals', logger.log_current_row['AverageQ2Vals'],t)
            writer.add_scalar('HuberLossQ', logger.log_current_row['LossQ'],t)
            writer.add_scalar('LearningRate', logger.log_current_row['LR'],t)

            logger.dump_tabular()


            