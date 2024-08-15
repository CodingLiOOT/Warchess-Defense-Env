import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym import spaces
from collections import deque
import random
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## PPO Policy ##################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_space):
        super(Actor, self).__init__()
        self.action_dims = action_space.nvec
        total_action_dim = sum(self.action_dims)
        self.split_sizes = tuple(self.action_dims)
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_action_dim)
        ).to(device)  # Ensure the network is on the correct device
    
    def forward(self, state):
        logits = self.network(state)
        split_logits = torch.split(logits, self.split_sizes, dim=-1)
        return [F.softmax(logit, dim=-1) for logit in split_logits]

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)  # Ensure the network is on the correct device
    
    def forward(self, state):
        return self.network(state)

# class Buffer:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.is_terminals = []

#     def clear(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.is_terminals = []
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, is_terminal):
        self.buffer.append((state, action, reward, is_terminal))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, K_epochs=4, batch_size=64, buffer_size=10000):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.num_positions = 40
        self.device = 'cuda:0'
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        probabilities = self.actor(state)
        # print(probabilities[0])
        # action = [p.multinomial(num_samples=1).item() for p in probabilities]
        # return np.array(action)
        mask = torch.ones(self.num_positions, dtype=torch.bool, device=self.device)
        actions = []

        # # for i in range(0, len(probabilities), 2):
        # #     p_position = probabilities[i].view(-1) * mask.float()
        # #     p_orientation = probabilities[i + 1].view(-1)
            
        # #     position = torch.multinomial(p_position, num_samples=1).item()
        # #     orientation = torch.multinomial(p_orientation, num_samples=1).item()
            
        # #     actions.append((position, orientation))
        # #     mask[position] = 0  # 禁用已选择的位置
        # mask[[0,1,3,4,5,6,8,9,10,11,13,14,15,16,21,22,17,18,23,24]] = 0
        # for i in range(0, 3):
        #     p_position = probabilities[i].view(-1) * mask.float()
        #     position = torch.multinomial(p_position, num_samples=1).item()
        #     actions.append(position)
        #     mask[position] = 0  # 禁用已选择的位置
        # mask[[0,1,3,4,5,6,8,9,10,11,13,14,15,16,21,22,17,18,23,24]] = 1
        # mask[[2, 7, 12, 19, 20]] = 0
        # for i in range(3, len(probabilities)):
        #     p_position = probabilities[i].view(-1) * mask.float()
        #     position = torch.multinomial(p_position, num_samples=1).item()
        #     actions.append(position)
        #     mask[position] = 0  # 禁用已选择的位置



        for i in range(12):
            offset = i*3

            p_position = probabilities[offset].view(-1) * mask.float()
            position = torch.multinomial(p_position, num_samples=1).item()

            angle_1_probs = probabilities[offset + 1]
            angle_1 = torch.multinomial(angle_1_probs, num_samples=1).item()

            angle_2_probs = probabilities[offset + 2]
            angle_2 = torch.multinomial(angle_2_probs, num_samples=1).item()

            # 将选定的动作添加到actions列表
            actions.append((position, angle_1, angle_2))

            mask[position] = 0  # 禁用已选择的位置

        # # 处理NearTurrets
        # for i in range(self.NearTurret_num):
        #     offset = i * 3

        #     # 获取位置的概率并应用掩码
        #     p_position = probabilities[offset].view(-1) * mask.float()
        #     position = torch.multinomial(p_position, num_samples=1).item()

        #     # 获取角度1的概率并采样
        #     angle_1_probs = probabilities[offset + 1].view(-1)
        #     angle_1 = torch.multinomial(angle_1_probs, num_samples=1).item()

        #     # 获取角度2的概率并采样
        #     angle_2_probs = probabilities[offset + 2].view(-1)
        #     angle_2 = torch.multinomial(angle_2_probs, num_samples=1).item()

        #     # 将选定的动作添加到actions列表
        #     actions.append((position, angle_1, angle_2))

        #     # 禁用已选择的位置
        #     mask[position] = 0

        # # 处理FarTurrets
        # offset = self.NearTurret_num * 3
        # for i in range(self.FarTurret_num):
        #     turret_offset = offset + i * 3

        #     # 获取位置的概率并应用掩码
        #     p_position = probabilities[turret_offset].view(-1) * mask.float()
        #     position = torch.multinomial(p_position, num_samples=1).item()

        #     # 获取角度1的概率并采样
        #     angle_1_probs = probabilities[turret_offset + 1].view(-1)
        #     angle_1 = torch.multinomial(angle_1_probs, num_samples=1).item()

        #     # 获取角度2的概率并采样
        #     angle_2_probs = probabilities[turret_offset + 2].view(-1)
        #     angle_2 = torch.multinomial(angle_2_probs, num_samples=1).item()

        #     # 将选定的动作添加到actions列表
        #     actions.append((position, angle_1, angle_2))

        #     # 禁用已选择的位置
        #     mask[position] = 0
        print(f'actions:{np.array(actions)}')

        return np.array(actions)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # 获取一批数据
        batch = self.buffer.sample(self.batch_size)
        states, actions, raw_rewards, is_terminals = zip(*batch)

        # 计算折扣奖励
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(raw_rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # 转换为张量并移至设备
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        states_tensor = torch.tensor(np.vstack(states), dtype=torch.float32).detach().to(device)
        actions_tensor = torch.tensor(np.vstack(actions), dtype=torch.int32).detach().to(device)

        # 使用PPO进行优化
        for _ in range(self.K_epochs):
            # 从评论家模型获取状态值
            state_values = self.critic(states_tensor)
            advantages = rewards_tensor - state_values.squeeze(1)  # Detach to avoid unwanted backprop
            
            actor_loss = -torch.mean(advantages)
            critic_loss = F.mse_loss(state_values.squeeze(1), rewards_tensor)
            
            # 更新演员网络
            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer_actor.step()
        
            # 更新评论家网络
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
            
        self.buffer.clear()


    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        
    def load(self, actor_filename, critic_filename):
        self.actor.load_state_dict(torch.load(actor_filename))
        self.critic.load_state_dict(torch.load(critic_filename))