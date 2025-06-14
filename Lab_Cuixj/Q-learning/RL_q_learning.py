from collections import defaultdict
import numpy as np
import pandas as pd


class QLearning:
    def __init__(self, states, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.states = states # s list
        self.actions = actions  # a list, a = [0,1,2,3]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        ''' build q table'''
        ############################

        # YOUR IMPLEMENTATION HERE #
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        ############################

    def choose_action(self, observation, maze_h, maze_w):
        ''' choose action from q table '''
        ############################

        # YOUR IMPLEMENTATION HERE #
        # epsilon greedy, on policy
        if np.random.uniform() < (1 - self.epsilon * (len(self.actions) - 1) / len(self.actions)):
            # 选择概率最大的动作
            state_action = self.q_table[observation]
            action = np.argmax(state_action)
        else:
            state_action = self.q_table[observation]
            max_index = np.argmax(state_action)
            action = np.random.choice(len(state_action))
            # 选择除去概率最大的动作之外的动作，且排除不合理动作
            while max_index == action or self.check_action(observation, action, maze_h, maze_w) == False:
                action = np.random.choice(len(state_action))
        return action

        ############################

    def learn(self, s, a, r, s_):
        ''' update q table '''
        ############################

        # YOUR IMPLEMENTATION HERE #

        self.q_table[s][a] = self.q_table[s][a] +\
                             self.lr * (r + self.gamma * np.max(self.q_table[s_]) - self.q_table[s][a])

        ############################

    def check_action(self, s, action, maze_h, maze_w):
        if action == 0:   # up
            if s[1] == 0:
                return False
        elif action == 1:   # down
            if s[1] == maze_h - 1:
                return False
        elif action == 2:   # right
            if s[0] == maze_w - 1:
                return False
        elif action == 3:   # left
            if s[0] == 0:
                return False
        return True

    def check_state_exist(self, state):
        ''' check state '''
        ############################

        # YOUR IMPLEMENTATION HERE #

        ############################