import time
import torch
from PPO import PPO  # 确保 PPO.py 文件中的类可以被正确导入
from game import RedBlueBattleEnv  # 确保 game.py 文件中的环境可以被正确导入
import numpy as np
from gym import spaces
import logging
import os
import time
import random
from torch.utils.tensorboard import SummaryWriter


current_time = time.strftime("%Y%m%d_%H%M%S",time.localtime())
# 设置保存目录路径
save_dir = f"logs/{current_time}_experiment"    
# 确保目录存在
os.makedirs(save_dir, exist_ok=True)
# 初始化 tensorboard
log_dir = os.path.join(save_dir,'log')
writer = SummaryWriter(log_dir=log_dir)

# 创建一个 logger
logger = logging.getLogger('wargame_logger')
logger.setLevel(logging.DEBUG)  # 设置最低捕获级别为 DEBUG

# 创建一个 file handler 专门处理写入日志文件
file_handler = logging.FileHandler(os.path.join(log_dir, 'training_log.log'), mode='w')
file_handler.setLevel(logging.DEBUG)  # 设置文件记录级别为 DEBUG
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 添加 handler 到 logger
logger.addHandler(file_handler)
# 初始化环境
env = RedBlueBattleEnv()

# 环境和 PPO 参数
# print(env.observation_space.shape)
# print(env.action_space.shape)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]  # 假设 action_space 已经适当定义

# deployment_points_num = 40 
# red_team_unit_num = 12
# angle_1 = 6
# angle_2 = 6  
# action_dim = spaces.MultiDiscrete([deployment_points_num,angle_1,angle_2] * red_team_unit_num)
# state_dim = 9*env.num_grids

deployment_points_num = 40 
NearTurret_num = 6
FarTurret_num = 6

angle_start = 6
Near_range = 5
Far_range = 6
action_dim = spaces.MultiDiscrete([deployment_points_num,angle_start,Near_range] * NearTurret_num + [deployment_points_num,angle_start,Far_range] * FarTurret_num)
state_dim = 9*env.num_grids

gamma = 0.99  # 折扣因子
K_epochs = 4  # PPO的更新次数
clip_ratio=0.2  # PPO的epsilon裁剪




# 设置训练设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# 初始化 PPO
model = PPO(map_channels=1,enemy_vector_dim=294+20, device=device,action_space=action_dim, gamma=gamma, K_epochs=K_epochs)

# 训练循环
num_episodes = 12800
rewards_per_episode = []
win_rates = []
action_list = []

for episode in range(num_episodes):
    # state = env.reset(random.randint(0,2))  # 重置环境
    state = env.reset(2)
    total_reward = 0
    # print(np.array(state.shape))
    done = False
    while not done:
        map_input = state['map']
        enemy_vector = state['enemy_triples']
        action,log_probs = model.select_action(map_input,enemy_vector)
        next_state, reward, done, result,reward_detail = env.step(action)
        # logger.debug(f'Episode {episode+1}, Action: {action}')
        total_reward += reward
        model.buffer.add(state, action, log_probs, reward, done)
        logger.debug(f'Episode {episode+1}, Reward: {reward}, Done: {done}, Winloss reward: {reward_detail[0]}, Blue dead reward: {reward_detail[1]}, Blue evacuated reward: {reward_detail[2]}, Red dead reward: {reward_detail[3]}')
        if result == 1:
            action_list.append(action)
        state = next_state
    
    rewards_per_episode.append(total_reward)
    win_rates.append(1 if result == 1 else 0)  # 胜利时记录1，否则记录0
    
    # logger.debug(f'Episode {episode+1}: Total Reward = {total_reward}, Win Rate = {win_rates[-1]}')
    if(episode+1)%10==0:
        window_size = 10
        averaged_win_rate = np.mean(win_rates[-window_size:])
        averaged_reward = np.mean(rewards_per_episode[-window_size:])
        writer.add_scalar('Average Win Rate', averaged_win_rate, episode + 1)
        writer.add_scalar('Average Reward', averaged_reward, episode + 1)
        
    if (episode + 1) % 64 == 0:
        model.update()  # 更新模型
        # print(f'Episode {episode + 1}: Total Reward = {total_reward}')
        logger.debug(f'Progress Update - Episode {episode+1}: Total Reward = {total_reward}')
    if(episode + 1) % 1000 == 0:
        model.save(save_dir,'ppo_model'+str(episode+1))
        # print("Model saved")
        logger.debug(f'Model saved after {episode+1} episodes')
    
        

# print(action_list)

# import matplotlib.pyplot as plt
# # 计算每1个episode的平均胜率
# window_size = 10
# averaged_win_rates = [np.mean(win_rates[i:i+window_size]) for i in range(0, len(win_rates), window_size)]
# averaged_rewards = [np.mean(rewards_per_episode[i:i+window_size]) for i in range(0, len(rewards_per_episode), window_size)]
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(range(len(averaged_win_rates)), averaged_win_rates, label='Average Win Rate')
# plt.title(f'Average Win Rate per {window_size} Episodes')
# plt.xlabel(f'Episode (in {window_size}s)')
# plt.ylabel('Average Win Rate')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(range(len(averaged_rewards)), averaged_rewards, color='red', label='Average Reward')
# plt.title(f'Average Reward per {window_size} Episodes')
# plt.xlabel(f'Episode (in {window_size}s)')
# plt.ylabel('Average Reward')
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(f"{timestamp}_result.png")
# plt.close()
#保存模型
model.save(save_dir,'ppo_model_final')
writer.close()
