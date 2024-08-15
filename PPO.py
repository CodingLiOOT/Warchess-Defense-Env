import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym import spaces
from collections import deque
import random
import os


################################## PPO Policy ##################################
class Encoder(nn.Module):
    def __init__(self, map_channels, enemy_vector_dim):
        super(Encoder, self).__init__()
        # CNN for map processing
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 45 * 33, 1024),
            nn.ReLU()
        )
        
        # Fully connected layers for enemy vector
        self.enemy_fc = nn.Sequential(
            nn.Linear(enemy_vector_dim, 512),
            nn.ReLU()
        )

    def forward(self, map_input, enemy_vector):
        map_features = self.map_cnn(map_input).squeeze(0)
        enemy_features = self.enemy_fc(enemy_vector)
        combined_features = torch.cat((map_features, enemy_features), dim=-1)
        return combined_features

class Actor(nn.Module):
    def __init__(self, encoder, action_space):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.action_dims = action_space.nvec
        total_action_dim = sum(self.action_dims)
        
        # Combined MLP for output
        self.final_mlp = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_action_dim)
        )
        
    def forward(self, map_input, enemy_vector):
        features = self.encoder(map_input, enemy_vector)
        logits = self.final_mlp(features)
        split_logits = torch.split(logits, tuple(self.action_dims), dim=-1)
        probabilities = torch.cat([F.softmax(logit, dim=-1) for logit in split_logits], dim=-1)
        return probabilities

class Critic(nn.Module):
    def __init__(self, encoder):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.network = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, map_input, enemy_vector):
        features = self.encoder(map_input, enemy_vector)
        return self.network(features)
    

# class Encoder(nn.Module):
#     def __init__(self, map_channels, enemy_vector_dim):
#         super(Encoder, self).__init__()
#         # CNN for map processing
#         self.map_cnn = nn.Sequential(
#             nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#             nn.Linear(64 * 45 * 33, 512),  # 减少全连接层的维度
#             nn.ReLU()
#         )
        
#         # Fully connected layers for enemy vector
#         self.enemy_fc = nn.Sequential(
#             nn.Linear(enemy_vector_dim, 256),  # 减少维度
#             nn.ReLU()
#         )

#     def forward(self, map_input, enemy_vector):
#         map_features = self.map_cnn(map_input).squeeze(0)
#         enemy_features = self.enemy_fc(enemy_vector)
#         combined_features = torch.cat((map_features, enemy_features), dim=-1)
#         return combined_features


# class Actor(nn.Module):
#     def __init__(self, encoder, action_space):
#         super(Actor, self).__init__()
#         self.encoder = encoder
#         self.action_dims = action_space.nvec
#         total_action_dim = sum(self.action_dims)
        
#         # Combined MLP for output with reduced layers
#         self.final_mlp = nn.Sequential(
#             nn.Linear(768, 256),  # 合理的隐藏层维度，以处理618维动作空间
#             nn.ReLU(),
#             nn.Linear(256, total_action_dim)
#         )
        
#     def forward(self, map_input, enemy_vector):
#         features = self.encoder(map_input, enemy_vector)
#         logits = self.final_mlp(features)
#         split_logits = torch.split(logits, tuple(self.action_dims), dim=-1)
#         probabilities = torch.cat([F.softmax(logit, dim=-1) for logit in split_logits], dim=-1)
#         return probabilities


# class Critic(nn.Module):
#     def __init__(self, encoder):
#         super(Critic, self).__init__()
#         self.encoder = encoder
#         self.network = nn.Sequential(
#             nn.Linear(768, 256),  # 与Actor保持一致的隐藏层维度
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        
#     def forward(self, map_input, enemy_vector):
#         features = self.encoder(map_input, enemy_vector)
#         return self.network(features)

    
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
    def __init__(self, map_channels,enemy_vector_dim, action_space,device, gamma=0.99, clip_ratio=0.2, K_epochs=4, batch_size=64, buffer_size=10000,epsilon=1.0,epsilon_min=0.01,epsilon_decay=0.995,is_train=True):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        
        self.device = device
        self.actor_encoder = Encoder(map_channels, enemy_vector_dim).to(self.device)
        self.critic_encoder = Encoder(map_channels, enemy_vector_dim).to(self.device)
        
        self.actor = Actor(self.actor_encoder, action_space).to(self.device)
        self.critic = Critic(self.critic_encoder).to(self.device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.num_positions = 40
        # self.device = 'cuda:0'
        if(is_train):
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
        else:
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0
        
        
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
    def select_action(self, map_input, enemy_vector):
        map_input = torch.FloatTensor(map_input).to(self.device)
        enemy_vector = torch.FloatTensor(enemy_vector).to(self.device)
        probabilities = self.actor(map_input, enemy_vector).squeeze(0)

        if probabilities.dim() == 0 or probabilities.size(0) != 618:
            raise ValueError("The output of the actor model is incorrect. Expected 618 probabilities, got {}".format(probabilities.size(0)))

        mask = torch.ones(40, dtype=torch.bool, device=self.device)  # 40个部署位置
        actions = []
        action_log_probs = []
        
        is_exploration = np.random.rand() < self.epsilon
        
        def process_turret(offset, range_dim):
            nonlocal mask

            # 处理位置动作
            if is_exploration:  # 探索
                valid_positions = torch.arange(40, device=self.device)[mask].cpu().numpy()
                if len(valid_positions) == 0:
                    raise ValueError("No valid positions available for selection.")
                position = np.random.choice(valid_positions)
                log_prob_position = torch.tensor(np.log(1.0 / len(valid_positions)), device=self.device, dtype=torch.float32)
            else:  # 利用
                p_position = probabilities[offset:offset + 40] * mask.float()
                dist_position = torch.distributions.Categorical(p_position)
                position = dist_position.sample().item()
                log_prob_position = dist_position.log_prob(torch.tensor(position, device=self.device, dtype=torch.long))
            
            action_log_probs.append(log_prob_position)
            mask[position] = 0  # 更新掩码，禁用选择的位置

            # 处理角度1动作
            angle_1_probs = probabilities[offset + 40:offset + 46]
            dist_angle_1 = torch.distributions.Categorical(angle_1_probs)
            angle_1 = dist_angle_1.sample().item()
            log_prob_angle_1 = dist_angle_1.log_prob(torch.tensor(angle_1, device=self.device, dtype=torch.long))
            action_log_probs.append(log_prob_angle_1)

            # 处理范围动作
            angle_2_probs = probabilities[offset + 46:offset + 46 + range_dim]
            dist_angle_2 = torch.distributions.Categorical(angle_2_probs)
            angle_2 = dist_angle_2.sample().item()
            log_prob_angle_2 = dist_angle_2.log_prob(torch.tensor(angle_2, device=self.device, dtype=torch.long))

            actions.append((position, angle_1, angle_2))
            action_log_probs.append(log_prob_angle_2)

        # 处理近程炮塔 (范围动作维度为 5)
        for i in range(6):  # 6个近程炮塔
            process_turret(i * 51, range_dim=5)

        # 处理远程炮塔 (范围动作维度为 6)
        for i in range(6):  # 6个远程炮塔
            process_turret(306 + i * 52, range_dim=6)  # 306是近程炮塔后的起始索引

        # 衰减 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        actions = np.array(actions)
        return actions, torch.stack(action_log_probs).detach().cpu().numpy()



    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # 从缓冲区采样
        batch = self.buffer.sample(self.batch_size)
        states, actions, old_log_probs, rewards, is_terminals = zip(*batch)
        
        # 分别提取 map 和 enemy_triples
        map_inputs = [state['map'] for state in states]
        enemy_vectors = [state['enemy_triples'] for state in states]
        
        # 将数据转换为 PyTorch 张量并移动到设备上
        map_inputs = torch.FloatTensor(np.vstack(map_inputs)).to(self.device)
        enemy_vectors = torch.FloatTensor(np.vstack(enemy_vectors)).to(self.device)
        actions = torch.LongTensor(np.vstack(actions)).to(self.device)  # actions 使用长整型
        old_log_probs = torch.FloatTensor(np.vstack(old_log_probs)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        # 标准化奖励
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        for _ in range(self.K_epochs):
            # 评估当前策略的 log_probs 和状态值
            log_probs, state_values = self.evaluate(map_inputs, enemy_vectors, actions)
            
            # 计算新旧策略的概率比率
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # 计算优势函数（advantages）
            advantages = rewards.unsqueeze(1) - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # 标准化优势

            # 计算 actor 损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算 critic 损失
            critic_loss = F.mse_loss(state_values, rewards.unsqueeze(1))  # 确保 rewards 形状匹配 state_values
            
            # 更新策略网络（actor）
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            # 更新价值网络（critic）
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # 清空缓冲区
        self.buffer.clear()


    def evaluate(self, map_inputs, enemy_vectors, actions):
        state_values = self.critic(map_inputs, enemy_vectors)
        probabilities = self.actor(map_inputs, enemy_vectors)
        log_probs = []

        def process_probabilities(probs, action_values, action_sizes):
            offset = 0
            for i in range(len(action_sizes)):
                dist = torch.distributions.Categorical(probs=probs[offset:offset + action_sizes[i]])
                action_value = action_values[i]

                # 确保 action_value 在 dist 的支持范围内
                if action_value >= action_sizes[i] or action_value < 0:
                    raise ValueError(f"Invalid action value {action_value} for distribution with size {action_sizes[i]}")

                log_prob = dist.log_prob(action_value)
                log_probs.append(log_prob.unsqueeze(0))
                offset += action_sizes[i]

        batch_size = probabilities.shape[0]

        for batch_index in range(batch_size):
            # 每个批次对应12个炮塔，每个炮塔有3个动作，因此取出对应的actions
            start_idx = batch_index * 12
            end_idx = start_idx + 12
            batch_actions = actions[start_idx:end_idx]

            # 近程炮塔的动作在前6个，远程炮塔的动作在后6个
            near_action_values = batch_actions[:6].reshape(-1, 3)  # 6个近程炮塔
            far_action_values = batch_actions[6:].reshape(-1, 3)   # 6个远程炮塔

            # Debugging: 打印切片后的 action_values 的形状和内容
            # print(f"near_action_values.shape: {near_action_values.shape}, near_action_values = {near_action_values}")
            # print(f"far_action_values.shape: {far_action_values.shape}, far_action_values = {far_action_values}")

            # 处理近程炮塔的概率
            near_probs = probabilities[batch_index, :306]  # 前 306 维度对应 6 个近程炮塔
            process_probabilities(near_probs, near_action_values.flatten(), [40, 6, 5])

            # 处理远程炮塔的概率
            far_probs = probabilities[batch_index, 306:]  # 后续维度对应 6 个远程炮塔
            process_probabilities(far_probs, far_action_values.flatten(), [40, 6, 6])

        total_log_probs = torch.cat(log_probs, dim=0).sum()
        return total_log_probs, state_values






    def save(self,save_dir, filename):
        torch.save(self.actor.state_dict(), os.path.join(save_dir,filename+"_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir,filename+"_critic.pth"))
        
    def load(self, actor_filename, critic_filename):
        self.actor.load_state_dict(torch.load(actor_filename))
        self.critic.load_state_dict(torch.load(critic_filename))
