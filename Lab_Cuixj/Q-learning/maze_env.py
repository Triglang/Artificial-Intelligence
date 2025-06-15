"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_q_learning.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
ORIGIN = [0,0]

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_states = MAZE_H * MAZE_W
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create 
        self.origin_index = ORIGIN
        self.origin = self.index2pixel(self.origin_index)

        # hell
        self.hell1_index = self.origin_index + np.array([2, 1])
        self.hell1_center = self.index2pixel(self.hell1_index)
        self.hell1 = self.canvas.create_rectangle(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black')
        # hell
        self.hell2_index = self.origin_index + np.array([1, 2])
        self.hell2_center = self.index2pixel(self.hell2_index)
        self.hell2 = self.canvas.create_rectangle(
            self.hell2_center[0] - 15, self.hell2_center[1] - 15,
            self.hell2_center[0] + 15, self.hell2_center[1] + 15,
            fill='black')

        # create oval
        self.oval_center_index = self.origin_index + np.array([2, 2])
        self.oval_center = self.index2pixel(self.oval_center_index)
        self.oval = self.canvas.create_oval(
            self.oval_center[0] - 15, self.oval_center[1] - 15,
            self.oval_center[0] + 15, self.oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.origin_index = ORIGIN
        self.origin = self.index2pixel(self.origin_index)
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')
        # return observation
        return self.origin_index

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 50
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -50
            done = False
        # 撞墙惩罚
        elif self.pixel2index(s)[1] == 0 and action == 0:
            reward = -1
            done = False
        elif self.pixel2index(s)[1] == MAZE_H - 1 and action == 1:
            reward = -1
            done = False
        elif self.pixel2index(s)[0] == MAZE_W - 1 and action == 2:
            reward = -1
            done = False
        elif self.pixel2index(s)[0] == 0 and action == 3:
            reward = -1
            done = False
        else:
            # reward = MAZE_H + MAZE_W - self.manhattanDist(s_, self.oval_center)
            # reward = self.manhattanDist(s, self.oval_center) - self.manhattanDist(s_, self.oval_center)
            reward = 0
            done = False

        # print(self.pixel2index(s_)[0:2])
        # print(reward)

        return self.pixel2index(s_)[0:2].astype(int).tolist(), reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

    @staticmethod
    def manhattanDist(s, s_):
        '''
        接收2个方框像素坐标，返回基于index的曼哈顿距离
        '''
        s = Maze.pixel2index(s)
        s_ = Maze.pixel2index(s_)
        return abs(s[0] - s_[0]) + abs(s[1] - s_[1])

    @staticmethod
    def index2pixel(index):
        return np.array(index) * UNIT + (UNIT / 2)
    
    @staticmethod
    def pixel2index(pixel):
        return np.array(pixel) // UNIT

def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()