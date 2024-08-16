import copy
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces

import utils


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
            nn.ReLU(),
        )

        # Fully connected layers for enemy vector
        self.enemy_fc = nn.Sequential(nn.Linear(enemy_vector_dim, 512), nn.ReLU())

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
            nn.Linear(128, total_action_dim),
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
            nn.Linear(128, 1),
        )

    def forward(self, map_input, enemy_vector):
        features = self.encoder(map_input, enemy_vector)
        return self.network(features)


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
    def __init__(
        self,
        map_channels,
        enemy_vector_dim,
        action_space,
        device,
        gamma=0.99,
        clip_ratio=0.2,
        K_epochs=4,
        batch_size=64 * 12,
        buffer_size=10000,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        is_train=True,
    ):
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

        indices = np.argwhere(np.array(utils.map) == 2)  # 获取(y, x)格式的索引
        indices_list = [tuple(index[::-1]) for index in indices]  # 转换为(x, y)格式并转为Python列表
        indices_list.sort()  # 排序
        self.deployment_points = [utils.xy2idx(x, y) for x, y in [x for x in indices_list]]

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.num_positions = 40
        # self.device = 'cuda:0'
        if is_train:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
        else:
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0

    def get_deployment_xy(self, pos):
        return utils.idx2xy(self.deployment_points[(int(pos))])

    def select_action(self, state):
        mask = {"dim1": torch.ones(40), "dim2": torch.ones(6), "dim3": torch.ones(6)}
        # 将 mask 字典中的每个张量移动到 MPS 设备上
        mask = {k: v.to(self.device) for k, v in mask.items()}
        actions = []
        buffer_tuple = []
        # 探索
        is_exploration = np.random.rand() < self.epsilon
        for step in range(12):
            turret_type = 'near' if step < 6 else 'far'
            action_dim1 = action_dim2 = action_dim3 = 0
            log_prob1 = log_prob2 = log_prob3 = 0.0
            if turret_type == 'near':
                mask['dim3'][-1] = 0
            if is_exploration:
                action_dist1 = torch.distributions.Categorical(mask['dim1'])
                action_dim1 = action_dist1.sample().item()
                log_prob1 = action_dist1.log_prob(
                    torch.tensor(action_dim1, device=self.device, dtype=torch.long)
                )
                action_dist2 = torch.distributions.Categorical(mask['dim2'])
                action_dim2 = action_dist2.sample().item()
                log_prob2 = action_dist2.log_prob(
                    torch.tensor(action_dim2, device=self.device, dtype=torch.long)
                )
                action_dist3 = torch.distributions.Categorical(mask['dim3'])
                action_dim3 = action_dist3.sample().item()
                log_prob3 = action_dist3.log_prob(
                    torch.tensor(action_dim3, device=self.device, dtype=torch.long)
                )
            else:
                map_input = torch.FloatTensor(state['map']).to(self.device)
                enemy_vector = torch.FloatTensor(state['enemy_triples']).to(self.device)
                probabilities = self.actor(map_input, enemy_vector).squeeze(0)
                logits1, logits2, logits3 = torch.split(probabilities, [40, 6, 6], dim=-1)
                # 第一维选择
                logits1 = logits1.cloned().detach()
                logits1[mask['dim1'] == 0] = -float('inf')
                action_dist1 = torch.distributions.Categorical(logits=logits1)
                action_dim1 = action_dist1.sample().item()
                log_prob1 = action_dist1.log_prob(
                    torch.tensor(action_dim1, device=self.device, dtype=torch.long)
                )
                # 第二维选择
                logits2 = logits2.cloned().detach()
                logits2[mask['dim2'] == 0] = -float('inf')
                action_dist2 = torch.distributions.Categorical(logits=logits2)
                action_dim2 = action_dist2.sample().item()
                log_prob2 = action_dist2.log_prob(
                    torch.tensor(action_dim2, device=self.device, dtype=torch.long)
                )
                # 第三维选择
                logits3 = logits3.cloned().detach()
                logits3[mask['dim3'] == 0] = -float('inf')
                action_dist3 = torch.distributions.Categorical(logits=logits3)
                action_dim3 = action_dist3.sample().item()
                log_prob3 = action_dist3.log_prob(
                    torch.tensor(action_dim3, device=self.device, dtype=torch.long)
                )
            actions.append(np.array([action_dim1, action_dim2, action_dim3]))
            state_copy = copy.deepcopy(state)
            buffer_tuple.append(
                (
                    state_copy,
                    (action_dim1, action_dim2, action_dim3),
                    torch.stack([log_prob1, log_prob2, log_prob3]).detach().cpu().numpy(),
                    0,
                    False,
                )
            )

            reshaped_arr = state['enemy_triples'][-120:].reshape(40, 3)
            x, y = self.get_deployment_xy(action_dim1)
            condition = (reshaped_arr[:, 0] == x) & (reshaped_arr[:, 1] == y)
            reshaped_arr[condition, 2] = 1 if turret_type == 'near' else 2
            state['enemy_triples'][-120:] = reshaped_arr.reshape(-1)
            mask['dim1'][action_dim1] = 0
        return actions, buffer_tuple

    def select_action_backup(self, map_input, enemy_vector):
        map_input = torch.FloatTensor(map_input).to(self.device)
        enemy_vector = torch.FloatTensor(enemy_vector).to(self.device)
        probabilities = self.actor(map_input, enemy_vector).squeeze(0)

        if probabilities.dim() == 0 or probabilities.size(0) != 618:
            raise ValueError(
                "The output of the actor model is incorrect. Expected 618 probabilities, got {}".format(
                    probabilities.size(0)
                )
            )

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
                log_prob_position = torch.tensor(
                    np.log(1.0 / len(valid_positions)),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:  # 利用
                p_position = probabilities[offset : offset + 40] * mask.float()
                dist_position = torch.distributions.Categorical(p_position)
                position = dist_position.sample().item()
                log_prob_position = dist_position.log_prob(
                    torch.tensor(position, device=self.device, dtype=torch.long)
                )

            action_log_probs.append(log_prob_position)
            mask[position] = 0  # 更新掩码，禁用选择的位置

            # 处理角度1动作
            angle_1_probs = probabilities[offset + 40 : offset + 46]
            dist_angle_1 = torch.distributions.Categorical(angle_1_probs)
            angle_1 = dist_angle_1.sample().item()
            log_prob_angle_1 = dist_angle_1.log_prob(
                torch.tensor(angle_1, device=self.device, dtype=torch.long)
            )
            action_log_probs.append(log_prob_angle_1)

            # 处理范围动作
            angle_2_probs = probabilities[offset + 46 : offset + 46 + range_dim]
            dist_angle_2 = torch.distributions.Categorical(angle_2_probs)
            angle_2 = dist_angle_2.sample().item()
            log_prob_angle_2 = dist_angle_2.log_prob(
                torch.tensor(angle_2, device=self.device, dtype=torch.long)
            )

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
        batch_size = probabilities.shape[0]
        log_probs = []

        for batch_index in range(batch_size):
            turret_actions = actions[batch_index]
            probs = probabilities[batch_index]

            dist1 = torch.distributions.Categorical(probs[:40])
            log_probs1 = dist1.log_prob(turret_actions[0])
            dist2 = torch.distributions.Categorical(probs[40:46])
            log_probs2 = dist2.log_prob(turret_actions[1])
            dist3 = torch.distributions.Categorical(probs[46:52])
            log_probs3 = dist3.log_prob(turret_actions[2])

            log_probs.append(log_probs1 + log_probs2 + log_probs3)
        total_log_probs = torch.stack(log_probs, dim=0).sum()
        return total_log_probs, state_values

    def evaluate_backup(self, map_inputs, enemy_vectors, actions):
        state_values = self.critic(map_inputs, enemy_vectors)
        probabilities = self.actor(map_inputs, enemy_vectors)
        log_probs = []

        def process_probabilities(probs, action_values, action_sizes):
            offset = 0
            for i in range(len(action_sizes)):
                dist = torch.distributions.Categorical(probs=probs[offset : offset + action_sizes[i]])
                action_value = action_values[i]

                # 确保 action_value 在 dist 的支持范围内
                if action_value >= action_sizes[i] or action_value < 0:
                    raise ValueError(
                        f"Invalid action value {action_value} for distribution with size {action_sizes[i]}"
                    )

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
            far_action_values = batch_actions[6:].reshape(-1, 3)  # 6个远程炮塔

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

    def save(self, save_dir, filename):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, filename + "_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, filename + "_critic.pth"))

    def load(self, actor_filename, critic_filename):
        self.actor.load_state_dict(torch.load(actor_filename))
        self.critic.load_state_dict(torch.load(critic_filename))
