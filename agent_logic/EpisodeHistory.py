import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import warnings

import contributors.Shazex_workspace.ShazexMain
import contributors.NotHappyDyadik_workspace.NotHappyDyadikMain
import contributors.DimaKovalchuk_workspace.DimaKovalchukMain
import contributors.eXce1z0r_workspace.eXce1z0rMain

warnings.filterwarnings("ignore")

team_name = 'ml_team # 1' # TODO: change your team name


class EpisodeHistory:
    def __init__(self,env,
                 learn_rate,
                 discount,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100
                 ):
        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.lr = learn_rate
        self.df = discount

        self.lvl_step = env.unwrapped.walls_per_level
        self.lvl_num = env.unwrapped.levels
        self.difficulty = env.unwrapped.level_difficulty
        self.point_plot = None
        self.mean_plot = None
        self.level_plots = []
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History. Team {}".format(team_name))

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)
        self.ax.set_title("Episode Length History (lr {}, df {})".format(self.lr, self.df))
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")
        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")
        for i in range(0, self.lvl_num):
            self.level_plots.append(plt.plot([],[], linewidth =1.0, c="#207232",ls ='--'))

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update levels plots
        for i in range(1, self.lvl_num+1):
            xl = range(plot_left_edge, plot_right_edge)
            yl = np.zeros(len(xl))
            yl[:] = i * self.lvl_step
            cur_lvl_curve = self.level_plots[i - 1][0]
            cur_lvl_curve.set_xdata(xl)
            cur_lvl_curve.set_ydata(yl)
            self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        #mean_kernel_size = 101
        #rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        #rolling_means = pd.rolling_mean(
        #    rolling_mean_data,
        #    window=mean_kernel_size,
        #    min_periods=0
        #)[mean_kernel_size:]
        #self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        #self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)

    def is_goal_reached(self, episode_index):
        ''' DO NOT CHANGE THIS FUNCTION CODE.'''
        # From here agent will receive sygnal about end of learning
        arr = self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1]
        avg = np.average(arr)
        if self.difficulty == 'Easy':
            answer = avg >= self.goal_avg_episode_length + 0.5
        elif len(arr)>0:
            density = 2 * np.max(arr) * np.min(arr) / (np.max(arr) + np.min(arr))
            answer = avg >= self.goal_avg_episode_length + 0.5 and density >= avg

        return answer