import argparse
import gymnasium as gym
from argument import dqn_arguments
import os
from agent_dir.agent_dqn import AgentDQN
from tqdm import tqdm

parser = argparse.ArgumentParser(description="dqn for cartpole")
parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
parser.add_argument('--test', action='store_true', help='whether test model')
parser = dqn_arguments(parser)

def do_train(args):
    env_name = args.env_name
    env = gym.make(env_name)
    agent = AgentDQN(env, args)
    agent.train()

def do_test(args):
    env_name = args.env_name
    env = gym.make(env_name, render_mode='human')
    agent = AgentDQN(env, args)
    agent.init_game_setting()
    done = False
    total_reward = 0
    q_bar = tqdm(desc='cur frame')
    s, _ = env.reset()
    while not done:
        q_bar.update(1)
        env.render()
        a = agent.make_action(s, test=True)
        s, r, done, truncated, _ = env.step(a)
        total_reward += r
        # print(done, truncated)
        if done or truncated:
            break
    env.close()
    print('total reward for test: {:.4f}'.format(total_reward))

if __name__ == '__main__':
    args = parser.parse_args()
    if args.train_dqn:
        print(args)
        do_train(args)
    if args.test:
        do_test(args)
