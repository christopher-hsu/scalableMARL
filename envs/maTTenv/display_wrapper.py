"""
edited by christopher-hsu from coco66 for multi_agent
"""
from gym import Wrapper
import numpy as np
from numpy import linalg as LA

import pdb, os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
from maTTenv.metadata import *

class Display2D(Wrapper):
    def __init__(self, env, figID = 0, skip = 1, confidence=0.95):
        super(Display2D,self).__init__(env)
        self.figID = figID # figID = 0 : train, figID = 1 : test
        """ If used in run_maTracking to debug: self.env_core = env
            if used in visualize_ma for normal use: self.env_core = env.env"""
        self.env_core = env.env
        self.bin = self.env_core.MAP.mapres
        if self.env_core.MAP.map is None:
            self.map = np.zeros(self.env_core.MAP.mapdim)
        else:
            self.map = self.env_core.MAP.map
        self.mapmin = self.env_core.MAP.mapmin
        self.mapmax = self.env_core.MAP.mapmax
        self.mapres = self.env_core.MAP.mapres
        self.fig = plt.figure(self.figID)
        self.n_frames = 0 
        self.skip = skip
        self.c_cf = np.sqrt(-2*np.log(1-confidence))

    def pos2map(self, obs, sd):
        x = obs[:,0]
        y = obs[:,1]

    def close(self):
        plt.close(self.fig)

    def render(self, mode='empty', record=False, traj_num=0, batch_outputs=None):
        if not hasattr(self, 'traj'):
            raise ValueError('Must do a env.reset() first before calling env.render()')

        # num_agents = len(self.traj)
        num_agents = self.env_core.nb_agents
        if type(self.env_core.agents) == list:
            agent_pos = [self.env_core.agents[i].state for i in range(num_agents)]
        else:
            agent_pos = self.env_core.agents.state

        # num_targets = len(self.traj_y)
        num_targets = self.env_core.nb_targets
        if type(self.env_core.targets) == list:
            target_true_pos = [self.env_core.targets[i].state[:2] for i in range(num_targets)]
            target_b_state = [self.env_core.belief_targets[i].state for i in range(num_targets)] # state[3:5]
            target_cov = [self.env_core.belief_targets[i].cov for i in range(num_targets)]
        else:
            target_true_pos = self.env_core.targets.state[:,:2]
            target_b_state = self.env_core.belief_targets.state[:,:2]  # state[3:5]
            target_cov = self.env_core.belief_targets.cov 

        if self.n_frames%self.skip == 0:
            self.fig.clf()       
            ax = self.fig.subplots()
            im = None
            if mode == 'empty':
                im = ax.imshow(self.map, cmap='gray_r', origin='lower',
                    extent=[self.mapmin[0], self.mapmax[0], self.mapmin[1], self.mapmax[1]])

            for ii in range(num_agents):
                #agents positions
                ax.plot(agent_pos[ii][0], agent_pos[ii][1], marker=(3, 0, agent_pos[ii][2]/np.pi*180-90),
                    markersize=10, linestyle='None', markerfacecolor='b', markeredgecolor='b')
                ax.plot(self.traj[ii][0], self.traj[ii][1], 'b.', markersize=2)
                #agents sensor indicators
                sensor_arc = patches.Arc((agent_pos[ii][0], agent_pos[ii][1]), METADATA['sensor_r']*2, METADATA['sensor_r']*2, 
                    angle = agent_pos[ii][2]/np.pi*180, theta1 = -METADATA['fov']/2, theta2 = METADATA['fov']/2, facecolor='gray')
                ax.add_patch(sensor_arc)
                ax.plot([agent_pos[ii][0], agent_pos[ii][0]+METADATA['sensor_r']*np.cos(agent_pos[ii][2]+0.5*METADATA['fov']/180.0*np.pi)],
                    [agent_pos[ii][1], agent_pos[ii][1]+METADATA['sensor_r']*np.sin(agent_pos[ii][2]+0.5*METADATA['fov']/180.0*np.pi)],'k', linewidth=0.5)
                ax.plot([agent_pos[ii][0], agent_pos[ii][0]+METADATA['sensor_r']*np.cos(agent_pos[ii][2]-0.5*METADATA['fov']/180.0*np.pi)],
                    [agent_pos[ii][1], agent_pos[ii][1]+METADATA['sensor_r']*np.sin(agent_pos[ii][2]-0.5*METADATA['fov']/180.0*np.pi)],'k', linewidth=0.5)
                self.traj[ii][0].append(agent_pos[ii][0])
                self.traj[ii][1].append(agent_pos[ii][1])

            for jj in range(num_targets):
                ax.plot(self.traj_y[jj][0], self.traj_y[jj][1], 'r.', markersize=2)
                ax.plot(target_true_pos[jj][0], target_true_pos[jj][1], marker='o', markersize=5, 
                    linestyle='None', markerfacecolor='r', markeredgecolor='r')
                # Belief on target
                ax.plot(target_b_state[jj][0], target_b_state[jj][1], marker='o', markersize=10, 
                    linewidth=5, markerfacecolor='none', markeredgecolor='g')
                eig_val, eig_vec = LA.eig(target_cov[jj][:2,:2])
                belief_target = patches.Ellipse((target_b_state[jj][0], target_b_state[jj][1]), 
                            2*np.sqrt(eig_val[0])*self.c_cf, 2*np.sqrt(eig_val[1])*self.c_cf, 
                            angle = 180/np.pi*np.arctan2(eig_vec[0][1],eig_vec[0][0]) ,fill=True,
                            zorder=2, facecolor='g', alpha=0.5)
                ax.add_patch(belief_target)
                self.traj_y[jj][0].append(target_true_pos[jj][0])
                self.traj_y[jj][1].append(target_true_pos[jj][1])
            
            ax.set_xlim((self.mapmin[0], self.mapmax[0]))
            ax.set_ylim((self.mapmin[1], self.mapmax[1]))
            ax.set_aspect('equal','box')
            ax.grid()
            ax.set_title(' '.join([mode.upper(),': Trajectory',str(traj_num)]))

            if not record :
                plt.draw()
                plt.pause(0.0005)
        self.n_frames += 1 

    def reset(self, **kwargs):
        self.traj = [[[],[]]]*self.env_core.num_agents
        self.traj_y = [[[],[]]]*self.env_core.num_targets
        return self.env.reset(**kwargs)

class Video2D(Wrapper):
    def __init__(self, env, dirname = '', skip = 1, dpi=80, local_view=False):
        super(Video2D, self).__init__(env)
        self.local_view = local_view
        self.skip = skip
        self.moviewriter = animation.FFMpegWriter()
        fname = os.path.join(dirname, 'eval_%da%dt.mp4'%(env.nb_agents,env.nb_targets))
        self.moviewriter.setup(fig=env.fig, outfile=fname, dpi=dpi)
        if self.local_view:
            self.moviewriter0 = animation.FFMpegWriter()
            fname0 = os.path.join(dirname, 'train_local_%d.mp4'%np.random.randint(0,20))
            self.moviewriter0.setup(fig=env.fig0, outfile=fname0, dpi=dpi)
        self.n_frames = 0

    def render(self, *args, **kwargs):
        if self.n_frames % self.skip == 0:
        #if traj_num % self.skip == 0:
            self.env.render(record=True, *args, **kwargs)
        self.moviewriter.grab_frame()
        if self.local_view:
            self.moviewriter0.grab_frame()
        self.n_frames += 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

