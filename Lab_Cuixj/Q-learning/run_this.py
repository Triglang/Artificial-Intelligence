"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
"""

from maze_env import Maze, MAZE_H, MAZE_W
from RL_q_learning import QLearning
from RL_sarsa import Sarsa


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            '''Renders policy once on environment. Watch your agent play!'''
            env.render()

            '''
            RL choose action based on observation
            e.g: action = RL.choose_action(str(observation))
            '''
            ############################

            # YOUR IMPLEMENTATION HERE #

            action = RL.choose_action(str(observation), MAZE_H, MAZE_W)
            # print(observation, 'action:', action)

            ############################

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            '''
            RL learn from this transition
            e.g: RL.learn(str(observation), action, reward, str(observation_), is_lambda=True)
                 RL.learn(str(observation), action, reward, str(observation_), is_lambda_return=True)
            '''
            ############################

            # YOUR IMPLEMENTATION HERE #

            RL.learn(str(observation), action, reward, str(observation_))

            ############################

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

        for state, action in RL.q_table.items():
            print(state, action)
        print('------')

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()

    '''
    build RL Class
    RL = QLearning(actions=list(range(env.n_actions)))
    RL = Sarsa(actions=list(range(env.n_actions)))
    '''
    ############################

    # YOUR IMPLEMENTATION HERE #
    RL = QLearning(states=list(range(env.n_states)), actions=list(range(env.n_actions)))

    ############################

    env.after(100, update)
    env.mainloop()

    for state, action in RL.q_table.items():
        print(state, action)