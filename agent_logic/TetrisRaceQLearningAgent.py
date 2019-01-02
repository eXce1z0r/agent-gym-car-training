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


class TetrisRaceQLearningAgent:
    def __init__(self,env,learning_rate = 0.5, discount_factor =0.5,
                 exploration_rate =0.5, exploration_decay_rate =0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.actions = env.unwrapped.actions
        self._num_actions = len(self.actions)
        self.state = None
        self.action = None

        # =============== TODO: Your code here ===============
        #  We'll use tabular Q-Learning in our agent, which means
        #  we need to allocate a Q - Table.Think about what exactly
        #  can be represented as states and how many states should
        #  be at all. Besides this remember about actions, which our
        #  agent will do. Q - table must contain notion about the state,
        #  represented as one single integer value (simplest option) and weights
        #  of each action, which agent can do in current env.

        self.wall_iterator = env.unwrapped.wall_iterator # passed walls counter

        #self.q_table = None
        self.q_table = [0 for i in range(1)]
        self.q_table[0] = [0, [[0, 0, 0, [0, 0, 0]]]] # q_table[car_y_pos, [[car_x_pos, left_turn_q_rate, right_turn_q_rate, [prev_car_y_pos, prev_car_x_pos, action]], ...]]
        del self.q_table[0]
#        for i in range(0, len(self.q_table)):
#            for j in range(0, len(self.q_table[0])):
#                self.q_table[i][j] = 0


    def choose_action(self, observation): # observation[0] - car x pos. observation[1] - car rewards. observation[2] - car y pos
        # =============== TODO: Your code here ===============
        #  Here agent must choose action on each step, solving exploration-exploitation
        #  trade-off. Remember that in general exploration rate is responsible for
        #  agent behavior in unknown world conditions and main motivation is explore world.
        #  Exploitation rate - choose already known actions and moving through known states.
        #  Think about right proportion that parameters for better solution
        self.check_state_exist(observation)

        #print('self.q_table: ', self.q_table)
        print(observation)

        action = -2
        for i in range(0, len(self.q_table)):
            if self.q_table[i][0] == observation[2]:
                for j in range(0, len(self.q_table[i][1])):
                    if self.q_table[i][1][j][0] == observation[0]:
                        if self.q_table[i][1][j][1] > self.q_table[i][1][j][2]:
                            action = 0
                        elif self.q_table[i][1][j][2] > self.q_table[i][1][j][1]:
                            action = 1
                        break
                break
        if action == -2:
            action = np.random.choice(self.actions)

        return action

    def act(self, state, action, reward, state_):
        # =============== TODO: Your code here ===============
        #  Here agent takes action('moves' somewhere), knowing
        #  the value of Q - table, corresponds current state.
        #  Also in each step agent should note that current
        #  'Q-value' can become 'better' or 'worsen'. So,
        #   an agent can update knowledge about env, updating Q-table.
        #   Remember that agent should choose max of Q-value in  each step

        self.check_state_exist(state_)

        for i in range(0, len(self.q_table)): # this loop used for saving info(car_y, car_x, action) about prev car turn
            if self.q_table[i][0] == state_[2]: # looking for same self.q_table y value and state_ y value
                for j in range(0, len(self.q_table[i][1])):
                    if self.q_table[i][1][j][0] == state_[0]: # looking for same self.q_table x value and x value which stores at state_ x states array
                        self.q_table[i][1][j][3][0] = state[2]
                        self.q_table[i][1][j][3][1] = state[0]
                        self.q_table[i][1][j][3][2] = action

                        print('state change info')
                        print('\told state: ', state)
                        print('\taction: ', action)
                        print('\tnew state: ', state_)
                        print('\treward: ', reward)
                        print('\tcurr_q_table_val: ', self.q_table[i])

                        for k in range(0, len(self.q_table)):
                            if self.q_table[k][0] == state[2]:
                                for l in range(0, len(self.q_table[k][1])):
                                    if self.q_table[k][1][l][0] == state[0]:
                                        self.recalcRewards(k, l, action, reward)
                                        break
                        break

        #self.q_table = None



        return self.q_table

    def recalcRewards(self, car_prev_y_index, car_prev_x_index, action, reward): # recalculates reward for each step from the current waypoint to the first

        #set reward from calc prev(next) step
        if action == 0: # reward for move left
            self.q_table[car_prev_y_index][1][car_prev_x_index][1] = reward
        elif action == 1: # reward for move right
            self.q_table[car_prev_y_index][1][car_prev_x_index][2] = reward

        #some print
        print('\t\tprev y state(recalc): ', self.q_table[car_prev_y_index])
        print('\t\tprev x prev state(recalc): ', self.q_table[car_prev_y_index][1][car_prev_x_index])

        #get data for q calc
        reward_left = self.q_table[car_prev_y_index][1][car_prev_x_index][1]
        reward_right = self.q_table[car_prev_y_index][1][car_prev_x_index][2]

        #calc new reward q rate
        q_l_reward = (reward_left + reward_right) * 0.9

        #find prev
        for k in range(0, len(self.q_table)):
            if self.q_table[k][0] == self.q_table[car_prev_y_index][1][car_prev_x_index][3][0]:
                for l in range(0, len(self.q_table[k][1])):
                    if self.q_table[k][1][l][0] == self.q_table[car_prev_y_index][1][car_prev_x_index][3][1]:
                        prev_action = self.q_table[car_prev_y_index][1][car_prev_x_index][3][2]
                        self.recalcRewards(k, l, prev_action, q_l_reward)
                        break

        pass

    def check_state_exist(self, state):
        # =============== TODO: Your code here ===============
        #  Here agent can write to Q-table new data vector if current
        #  state is unknown for the moment

        #if state[2] == 188.0:
        #    print("GOTCHA!")

        existedYIndex = -1
        isXExist = True
        for i in range(0, len(self.q_table)):
            if self.q_table[i][0] == state[2]:
        #        if state[2] == 188.0:
        #            print("GOTCHA v2!")
                existedYIndex = i
                for j in range(0, len(self.q_table[i][1])):
                    if self.q_table[i][1][j][0] == state[0]:
                        isXExist = False
                        break
                break


        if existedYIndex == -1:
        #    if state[2] == 188.0:
        #        print("GOTCHA v3!")
            self.q_table.append([state[2], [[0, 0, 0, [0, 0, 0]]]])
            self.q_table[len(self.q_table)-1][1][0][0] = state[0]
        #elif not isXExist:
        elif isXExist:
        #    if state[2] == 188.0:
        #        print("GOTCHA v4!")
        #    if state[2] == 188.0:
        #        print('existedYIndex: ', existedYIndex)
            self.q_table[existedYIndex][1].append([state[0], 0, 0])

        self.q_table.sort()

        pass