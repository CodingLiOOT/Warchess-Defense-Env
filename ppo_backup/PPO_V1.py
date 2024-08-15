import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym import spaces

################################## set device ##################################
# print("============================================================================================")
# # set device to cpu or cuda
# device = torch.device('cpu')
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")


################################## PPO Policy ##################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_space):
        super(Actor, self).__init__()
        self.action_dims = action_space.nvec
        total_action_dim = sum(self.action_dims)

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_action_dim),
        )
        self.split_sizes = tuple(self.action_dims)

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
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.network(state)


class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []


class PPO:
    def __init__(
        self, state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = Buffer()

        self.actor = Actor(state_dim, action_space)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        probabilities = self.actor(state)
        action = [p.multinomial(num_samples=1).item() for p in probabilities]
        return np.array(action)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_states = torch.tensor(np.vstack(self.buffer.states), dtype=torch.float32)
        old_actions = torch.tensor(np.vstack(self.buffer.actions), dtype=torch.int32)

        for _ in range(self.K_epochs):
            # This is a simplified version of PPO for clarity. Add clipping and entropy terms as needed.
            state_values = self.critic(old_states)
            advantages = rewards - state_values.squeeze(1)

            actor_loss = -torch.mean(advantages)  # A placeholder for proper calculation
            critic_loss = F.mse_loss(state_values.squeeze(1), rewards)

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        self.buffer.clear()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
