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
        probabilities = torch.cat([F.softmax(logit, dim=-1) for logit in split_logits], dim=-1)
        return probabilities


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
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, log_probs, reward, is_terminal):
        self.buffer.append((state, action, log_probs, reward, is_terminal))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)	
    
class PPO:
    def __init__(self, state_dim, action_space, gamma=0.99, clip_ratio=0.2, K_epochs=4, batch_size=64, buffer_size=10000):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.actor = Actor(state_dim, action_space).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.num_positions = 40
        self.device = 'cuda:0'
        
        
    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     probabilities = self.actor(state)
    #     # print(probabilities[0])
    #     # action = [p.multinomial(num_samples=1).item() for p in probabilities]
    #     # return np.array(action)
    #     mask = torch.ones(self.num_positions, dtype=torch.bool, device=self.device)
    #     actions = []
    #     action_log_probs = []
    #     for i in range(12):
    #         offset = i*3

    #         p_position = probabilities[offset].view(-1) * mask.float()
    #         position = torch.multinomial(p_position, num_samples=1).item()

    #         angle_1_probs = probabilities[offset + 1]
    #         angle_1 = torch.multinomial(angle_1_probs, num_samples=1).item()

    #         angle_2_probs = probabilities[offset + 2]
    #         angle_2 = torch.multinomial(angle_2_probs, num_samples=1).item()

    #         # 将选定的动作添加到actions列表
    #         actions.append((position, angle_1, angle_2))

    #         mask[position] = 0  # 禁用已选择的位置
    #     print(f'actions:{np.array(actions)}')

    #     return np.array(actions)
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        probabilities = self.actor(state).squeeze(0)

        if probabilities.dim() == 0 or probabilities.size(0) != 618:
            raise ValueError("The output of the actor model is incorrect. Expected 618 probabilities, got {}".format(probabilities.size(0)))

        mask = torch.ones(40, dtype=torch.bool, device=self.device)  # 40个部署位置
        actions = []
        action_log_probs = []

        # 先处理近程炮塔
        for i in range(6):  # 6个近程炮塔
            offset = i * 51
            # 处理位置动作
            p_position = probabilities[offset:offset + 40] * mask.float()
            dist_position = torch.distributions.Categorical(p_position)
            position = dist_position.sample()
            log_prob_position = dist_position.log_prob(position)
            action_log_probs.append(log_prob_position)
            # 处理角度1动作
            angle_1_probs = probabilities[offset + 40:offset + 46]
            dist_angle_1 = torch.distributions.Categorical(angle_1_probs)
            angle_1 = dist_angle_1.sample()
            log_prob_angle_1 = dist_angle_1.log_prob(angle_1)
            action_log_probs.append(log_prob_angle_1)
            # 处理范围动作
            angle_2_probs = probabilities[offset + 46:offset + 51]
            dist_angle_2 = torch.distributions.Categorical(angle_2_probs)
            angle_2 = dist_angle_2.sample()
            log_prob_angle_2 = dist_angle_2.log_prob(angle_2)

            actions.append((position.item(), angle_1.item(), angle_2.item()))
            action_log_probs.append(log_prob_angle_2)

            # 更新位置掩码
            mask[position] = 0

        # 处理远程炮塔
        for i in range(6):  # 6个远程炮塔
            offset = 306 + i * 52  # 306是近程炮塔后的起始索引
            # 处理位置动作
            p_position = probabilities[offset:offset + 40] * mask.float()
            dist_position = torch.distributions.Categorical(p_position)
            position = dist_position.sample()
            log_prob_position = dist_position.log_prob(position)
            action_log_probs.append(log_prob_position)
            # 处理角度1动作
            angle_1_probs = probabilities[offset + 40:offset + 46]
            dist_angle_1 = torch.distributions.Categorical(angle_1_probs)
            angle_1 = dist_angle_1.sample()
            log_prob_angle_1 = dist_angle_1.log_prob(angle_1)
            action_log_probs.append(log_prob_angle_1)
            # 处理范围动作
            angle_2_probs = probabilities[offset + 46:offset + 52]
            dist_angle_2 = torch.distributions.Categorical(angle_2_probs)
            angle_2 = dist_angle_2.sample()
            log_prob_angle_2 = dist_angle_2.log_prob(angle_2)

            actions.append((position.item(), angle_1.item(), angle_2.item()))
            action_log_probs.append(log_prob_angle_2)

            # 更新位置掩码
            mask[position] = 0

        actions = np.array(actions)
        return actions, torch.stack(action_log_probs).detach().cpu().numpy()


    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, old_log_probs, rewards, is_terminals = zip(*batch)

        states = torch.FloatTensor(np.vstack(states)).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        print(actions)
        old_log_probs = torch.FloatTensor(np.vstack(old_log_probs)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        
        for _ in range(self.K_epochs):
            log_probs, state_values = self.evaluate(states, actions)
            if isinstance(log_probs, list):
                log_probs = torch.stack(log_probs)  # 将log_probs列表转换为张量
            # 确保log_probs形状正确
            if log_probs.dim() == 2 and log_probs.shape[1] == 1:
                log_probs = log_probs.squeeze(1)  # 去掉单一维度
            log_probs = log_probs.view_as(old_log_probs)  # 重塑形状匹配old_log_probs

            print(log_probs.shape, log_probs)
            print(old_log_probs.shape, old_log_probs)

            ratios = torch.exp(log_probs - old_log_probs.detach())

            advantages = rewards.unsqueeze(1) - state_values.detach()
            
            print(f'Advantages = {advantages}')
            print(ratios.shape, ratios)
            print(rewards.shape, rewards)
            print(state_values.shape, state_values)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(state_values, rewards)
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        self.buffer.clear()

    def evaluate(self, states, actions):
        log_probs = []
        state_values = self.critic(states)
        probabilities = self.actor(states) #[2,618]



        # 由于概率是按批次组织的，我们需要处理每个批次的概率
        for batch_index in range(probabilities.shape[0]):  # 迭代每个批次
            # 近程炮塔动作处理
            near_probs = probabilities[batch_index, :306]  # 近程炮塔占前306个概率
            offset = 0
            for i in range(6):  # 6个近程炮塔
                for j, size in zip(range(3), [40, 6, 5]):  # 三种动作的范围
                    dist = torch.distributions.Categorical(probs=near_probs[offset:offset + size])                                  
                    log_prob = dist.log_prob(actions[batch_index * 12 + i, j])
                    log_probs.append(log_prob.unsqueeze(0))
                    offset += size

            # 远程炮塔动作处理
            far_probs = probabilities[batch_index, 306:]  # 远程炮塔开始于第306个概率
            offset = 0
            for i in range(6):  # 6个远程炮塔
                for j, size in zip(range(3), [40, 6, 6]):  # 三种动作的范围
                    dist = torch.distributions.Categorical(probs=far_probs[offset:offset + size])
                    log_prob = dist.log_prob(actions[batch_index * 12 + i + 6, j])
                    log_probs.append(log_prob.unsqueeze(0))
                    offset += size
        print(log_probs)
        total_log_probs = torch.cat(log_probs).sum(dim=0)
        return log_probs, state_values



    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        
    def load(self, actor_filename, critic_filename):
        self.actor.load_state_dict(torch.load(actor_filename))
        self.critic.load_state_dict(torch.load(critic_filename))
