import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import warnings

import agent_logic.Regression as RegressionClass
import agent_logic.TetrisRaceQLearningAgent as TetrisRaceQLearningAgentClass
import agent_logic.EpisodeHistory as EpisodeHistoryClass
import agent_logic.Classification as ClassificationClass

import contributors.Shazex_workspace.ShazexMain
import contributors.NotHappyDyadik_workspace.NotHappyDyadikMain
import contributors.DimaKovalchuk_workspace.DimaKovalchukMain
import contributors.eXce1z0r_workspace.eXce1z0rMain

warnings.filterwarnings("ignore")

global team_name, folder, env_name
team_name = 'ml_team # 1' # TODO: change your team name
folder = 'tetris_race_qlearning'
env_name = 'TetrisRace-v0' # do not change this


class Controller:
    def __init__(self, parent_mode = True , episodes_num = 10000, global_env = []):
        self.team_name = team_name
        self.exp_dir = folder + '/' + self.team_name
        random_state = 0
        self.agent_history = []
        self.history_f = True

        self.window = 50

        if parent_mode == False:
            # ====== TODO: your code here======
            # To run env with different parameters you can use another values of named variables, such as:
            #
            # walls_num = x --> number of obstacles (walls). The number must be  x > 6 and x % 3 == 0
            # walls_spread = x --> distance between walls. Too small value leads to no solution situation
            # episodes_to_run = x --> number of agent's tries to learn
            # world_type = 'Fat' or 'Thin' --> make objects more thicker or thinner
            # smooth_car_step = 5 -->  smoothness of car moves (value of step by x)
            # level_difficulty ='Easy' or 'Medium' --> change number of bricks in walls
            # car_spawn = 'Random' or 'Center' --> place, where car starts go
            #
            # EXAMPLE:
            # env.__init__(walls_num = 6, walls_spread = 3, episodes_to_run = episodes_num)
            #
            # Best choice will try any of this different options for better understanding and
            # optimizing the solution.

            env = gym.make(env_name)

            env.__init__(walls_num=9, walls_spread=20, episodes_to_run=1000, car_spawn='Center')

            env.seed(random_state)
            np.random.seed(random_state)
            lr = 10
            df = 10
            exr = 10
            exrd = 10

            self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                            video_callable=self.video_callable)
            episode_history, end_index = self.run_agent(self, lr, df, exr, exrd, self.env,
                                                        verbose=False)
        else:
            # Here all data about env will received from main script, so
            # each team will work with equal initial conditions
            # ====== TODO: your code here======
            env = global_env
            env.seed(random_state)
            np.random.seed(random_state)

            self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                            video_callable=self.video_callable)
            episode_history, end_index = self.run_agent(self, self.learning_rate, self.discount_factor,
                                                        self.exploration_rate, self.exploration_decay_rate,
                                                        self.env, verbose=False)

    def run_agent(self, rate, factor, exploration, exp_decay, env, verbose=False):
        max_episodes_to_run = env.unwrapped.total_episodes
        max_timesteps_per_episode = env.unwrapped.walls_num

        goal_avg_episode_length = env.unwrapped.walls_num
        wall_coef = 6 / env.unwrapped.walls_num
        goal_consecutive_episodes = int(wall_coef * self.window)  # how many times agent can consecutive run succesful

        plot_episode_count = 200
        plot_redraw_frequency = 10

        # =============== TODO: Your code here ===============
        #   Create a Q-Learning agent with proper parameters.
        #   Think about what learning rate and discount factor
        #   would be reasonable in this environment.

        agent = TetrisRaceQLearningAgentClass.TetrisRaceQLearningAgent(env,
                                         learning_rate=rate,
                                         discount_factor=factor,
                                         exploration_rate=exploration,
                                         exploration_decay_rate=exp_decay
                                         )
        # ====================================================
        episode_history = EpisodeHistoryClass.EpisodeHistory(env,
                                         learn_rate=rate,
                                         discount=factor,
                                         capacity=max_episodes_to_run,
                                         plot_episode_count=plot_episode_count,
                                         max_timesteps_per_episode=max_timesteps_per_episode,
                                         goal_avg_episode_length=goal_avg_episode_length,
                                         goal_consecutive_episodes=goal_consecutive_episodes)
        episode_history.create_plot()

        finish_freq = [0.5, True]  # desired percent finishes in window, flag to run subtask once
        for episode_index in range(0, max_episodes_to_run):
            timestep_index = 0
            observation = env.reset()

            while True:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)  # Perform the action and observe the new state.

                action_reward = reward

                if verbose == True:
                    env.render()
                    # self.log_timestep(timestep_index, action, reward, observation)

                if done and timestep_index < max_timesteps_per_episode - 1:
                    reward = -max_episodes_to_run

                #QDF = agent.act(observation, action, reward, observation_)
                QDF = agent.act(observation, action, action_reward, observation_)

                observation = observation_

                if done:
                    self.done_manager(self, episode_index, [], [], 'D')
                    if self.done_manager(self, episode_index, [], finish_freq, 'S') and finish_freq[1]:
                        foo = ClassificationClass.Classification()
                        finish_freq[1] = False

                    episode_history[episode_index] = timestep_index + 1
                    if verbose or episode_index % plot_redraw_frequency == 0:
                        episode_history.update_plot(episode_index)

                    if episode_history.is_goal_reached(episode_index):
                        print("Goal reached after {} episodes!".format(episode_index + 1))
                        print("QDF: ", QDF)
                        end_index = episode_index + 1
                        foo = RegressionClass.Regression(QDF)
                        self.done_manager(self, [], plt, [], 'P')

                        return episode_history, end_index
                    break
                elif env.unwrapped.wall_iterator - timestep_index > 1:
                    timestep_index += 1
        print("Goal not reached after {} episodes.".format(max_episodes_to_run))
        print("QDF: ", QDF)

        end_index = max_episodes_to_run
        return episode_history, end_index

    def done_manager(self, episode_ind, plt, top, mode):
        # Call this function to handle episode end event and for storing some
        # result files, pictures etc

        if mode == 'D':  # work with history data
            refresh_each = 100
            self.agent_history.append(self.env.unwrapped.wall_iterator)
            if episode_ind % refresh_each == 0 and self.history_f:
                root = self.exp_dir.split('/')[0]
                base = '/_data'
                path = root + base
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(path + '/' + self.team_name + '.pickle', 'wb') as f:
                    pickle.dump(self.agent_history, f)
        if mode == 'P':  # work woth progress plot
            path = self.exp_dir + '/learn_curve'
            name = '/W ' + str(self.env.unwrapped.walls_num) + \
                   '_LR ' + str(self.learning_rate) + '_DF ' + str(self.discount_factor)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name + '.png')
        if mode == 'S':  # call subtasks when condition
            if episode_ind > self.window:
                arr = self.agent_history[episode_ind - self.window: episode_ind]
                mx = np.max(arr)
                ind = np.where(arr == mx)[0]
                count = ind.shape[0]
                prc = count / self.window if mx > self.env.unwrapped.walls_per_level * 2  else 0
                x = self.agent_history
                total_finishes = sum(map(lambda x: x > self.env.unwrapped.walls_per_level * 2, x))

                return prc >= top[0] and total_finishes > 100

    def video_callable(episode_id):
        # call agent draw eact N episodes
        return episode_id % 300 == 0

    def log_timestep(self, index, action, reward, observation):
        # print parameters in console
        format_string = "   ".join(['Timestep:{}',
                                    'Action:{}',
                                    'Reward:{}',
                                    'Car pos:{}',
                                    'WallY pos:{}'])
        print('Timestep: format string ', format_string.format(index, action, reward,
                                                               observation[0], observation[1]))

    def save_history(self, history, experiment_dir):
        # Save the episode lengths to CSV.
        filename = os.path.join(experiment_dir, "episode_history.csv")
        dataframe = pd.DataFrame(history.lengths, columns=["length"])
        dataframe.to_csv(filename, header=True, index_label="episode")

def main(env, parent_mode = True):
    obj = Controller
    obj.__init__(obj, parent_mode= parent_mode, global_env= env)

if __name__ == "__main__":
    if 'master.py' not in os.listdir('.'):
        main([],parent_mode=False)
    else:
        main(env, parent_mode)