import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
# from tensorboardX import SummaryWriter
from torch import nn, optim
# from agent_dir.agent_dqn import Agent
from collections import deque
from tqdm import tqdm
import logging

class QNetwork(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        out_dim,
        device,
        activation="relu",
        dropout=0.0
    ):
        """
        初始化带单隐藏层的Q网络
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            out_dim: 输出维度
            device: 计算设备
            activation: 激活函数类型
            dropout: dropout概率
        """
        super(QNetwork, self).__init__()
               
        # 保存参数
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'none': nn.Identity()
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.activation = activation_map[activation]
        
        # 构建单隐藏层网络
        layers = [
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim)
        ]
        
        # 组合成顺序模型
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, inputs):
        """前向传播"""
        # 确保输入在正确设备上
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        
        return self.net(inputs)
    
    # 更换计算设备
    def set_device(self, device: torch.device) -> torch.nn.Module:
        _model = self.to(device)
        _model.device = device
        return _model
    
    # 保存模型
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    # 加载模型
    def load(self, path: str):
        self.load_state_dict(torch.load(path))

from collections import deque, namedtuple
# 定义一个命名元组来表示经验
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])
class ReplayBuffer:
    def __init__(self, buffer_size):
        '''
        初始化经验回放缓冲区
        
        参数:
            buffer_size: 缓冲区的最大容量
        '''
        self.buffer_size = buffer_size  # 最大容量
        self.buffer = deque(maxlen=buffer_size)  # 使用双端队列实现，自动处理超限

    def __len__(self):
        return len(self.buffer)
    
    def full(self):
        return len(self.buffer) == self.buffer_size

    def push(self, *transition):
        '''
        向缓冲区添加一条新的经验
        参数格式: (state, action, reward, next_state, done)
        '''
        # 将经验转换为命名元组并添加到缓冲区
        # 元组解包顺序: state, action, reward, next_state, done
        exp = Experience(*transition)
        self.buffer.append(exp)

    def sample(self, batch_size):
        '''
        从缓冲区中随机采样一批经验
        返回: (states, actions, rewards, next_states, dones)
        '''
        # 如果请求的批量大小大于可用数据量，则使用所有可用数据
        actual_batch_size = min(batch_size, len(self.buffer))
        
        # 随机采样一批经验
        batch = random.sample(self.buffer, actual_batch_size)
        
        # 将经验拆分为单独的数组
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为NumPy数组以便后续处理
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )

    def clean(self):
        self.buffer.clear()
        
    def __repr__(self):
        '''返回缓冲区的简要描述'''
        return f"ReplayBuffer(size={len(self)}/{self.buffer_size})"

class AgentDQN:
    def __init__(self, env, args):
        """
        初始化DQN智能体
        
        参数:
            env: 环境对象
            args: 命令行参数
        """
        self.env = env
        self.args = args
        
        # 获取环境参数
        self.state_dim = env.observation_space.shape[0]     # 状态空间维度
        self.action_dim = env.action_space.n        # 动作空间大小
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建Q网络
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不更新参数
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.args.learning_rate
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.args.buffer_size)
        
        # 训练参数初始化
        self.epsilon = self.args.epsilon_start
        self.total_steps = 0
        self.episode_count = 0
        
        # 保存路径
        self.save_dir = self.args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _build_network(self):
        """构建Q网络"""
        return QNetwork(
            input_dim=self.state_dim,
            hidden_dim=self.args.hidden_size,
            out_dim=self.action_dim,
            device=self.device,
            activation=self.args.activation,
            dropout=self.args.dropout
        )
    
    def init_game_setting(self):
        """
        测试前初始化设置
        - 加载最佳模型
        - 禁用探索
        """
        self._load_model()
        self.epsilon = self.args.epsilon_min  # 测试时使用最小探索率

    def train(self):
        """DQN训练主循环"""
        print(f"Starting DQN training on {self.device}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # 进度条初始化
        episode_rewards = []
        progress_bar = tqdm(total=self.args.n_iter, desc="Training DQN")
        
        while self.total_steps < self.args.n_iter:
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            episode_reward = 0
            
            # 单个episode循环
            while not done:
                # 选择动作
                action = self._select_action(state, training=True)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)
                
                # 存储经验
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # 更新状态
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "episode": self.episode_count,
                    "reward": episode_reward,
                    "epsilon": self.epsilon,
                    "buffer": len(self.replay_buffer)
                })
                
                # 定时更新目标网络
                if self.total_steps % self.args.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # 执行训练步骤
                if len(self.replay_buffer) > self.args.batch_size:
                    self._update_network()
            
            # 记录并更新探索率
            episode_rewards.append(episode_reward)
            self._decay_epsilon()
            self.episode_count += 1
            
            # 保存模型
            if self.episode_count % self.args.save_interval == 0:
                self._save_model()
                avg_reward = np.mean(episode_rewards[-self.args.save_interval:])
                print(f"Episode {self.episode_count}: Avg reward = {avg_reward:.2f}")
        
        # 清理
        progress_bar.close()
        self._save_model()
        print("Training complete. Final model saved.")
    
    def _select_action(self, state, training=False):
        """
        ε-贪心策略选择动作
        
        参数:
            state: 当前状态
            training: 是否为训练模式
        """
        # 测试模式或超过探索概率阈值时选择最优动作
        if not training or random.random() > self.epsilon:
            state_tensor = torch.tensor([state], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
        
        # 否则随机探索
        return random.randrange(self.action_dim)
    
    def _decay_epsilon(self):
        """指数衰减探索率"""
        self.epsilon = max(
            self.args.epsilon_min,
            self.args.epsilon_decay * self.epsilon
        )
    
    def _update_network(self):
        """更新策略网络"""
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.args.batch_size
        )
        
        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # 计算当前Q值 (Q(s,a))
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值 (max_a' Q_target(s', a'))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.args.gamma * next_q * (1 - dones)
        
        # 计算Huber损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.args.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_clip)
        
        self.optimizer.step()
    
    def make_action(self, observation, test=False):
        """
        基于当前状态选择动作
        
        参数:
            observation: 当前观察到的状态
            test: 是否为测试模式
        """
        # 测试模式直接使用最优策略
        if test:
            with torch.no_grad():
                state = torch.tensor([observation], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
            return torch.argmax(q_values).item()
        
        # 训练模式使用ε-贪心策略
        return self._select_action(observation, training=True)
    
    def soft_update(self, tau):
        """
        软更新目标网络参数
        
        参数:
            tau: 软更新系数 (0-1), tau=1相当于硬更新
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )
    
    def _save_model(self):
        """保存模型"""
        policy_path = os.path.join(self.save_dir, "dqn_policy.pth")
        target_path = os.path.join(self.save_dir, "dqn_target.pth")
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)
    
    def _load_model(self):
        """加载模型"""
        policy_path = os.path.join(self.save_dir, "dqn_policy.pth")
        if os.path.exists(policy_path):
            self.policy_net.load(policy_path)
            self.target_net.load(policy_path)
            print(f"Loaded model from {policy_path}")
