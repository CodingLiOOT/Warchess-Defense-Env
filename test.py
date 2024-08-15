import torch
from PPO import PPO  # 确保 PPO.py 文件中的类可以被正确导入
from game import RedBlueBattleEnv  # 确保 game.py 文件中的环境可以被正确导入
import numpy as np
from gym import spaces
import logging
import os
import time
import random

# 设置日志路径，包含时间戳
log_directory = 'wargame_log/{}'.format(time.strftime("%y%m%d%H%M", time.localtime()))
os.makedirs(log_directory, exist_ok=True)

# 创建一个 logger
logger = logging.getLogger('wargame_logger')
logger.setLevel(logging.DEBUG)  # 设置最低捕获级别为 DEBUG

# 创建一个 file handler 专门处理写入日志文件
file_handler = logging.FileHandler(
    os.path.join(log_directory, 'testing_log.log'), mode='w'
)
file_handler.setLevel(logging.DEBUG)  # 设置文件记录级别为 DEBUG
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

# 添加 handler 到 logger
logger.addHandler(file_handler)

platform = "Mac"
if platform == "Mac":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(style, num_simulation, env):
    succes_cnt = 0
    for sim in range(num_simulation):
        if style == 3:
            state = env.reset(random.randint(0, 2))
        else:
            state = env.reset(style)
        done = False
        while not done:
            map_input = state['map']
            enemy_vector = state['enemy_triples']
            action, log_probs = model.select_action(map_input, enemy_vector)
            next_state, reward, done, result, reward_detail = env.step(action)
            if result == 1:
                succes_cnt += 1
    return succes_cnt
    # print("Action:", action)
    # logger.debug(f'Episode {sim+1}, Action: {action}')
    # model.buffer.add(state, action, log_probs, reward, done)
    # logger.debug(f'Episode {sim+1}, Reward: {reward}, Done: {done}, Winloss reward: {reward_detail[0]}, Blue dead reward: {reward_detail[1]}, Blue evacuated reward: {reward_detail[2]}, Red dead reward: {reward_detail[3]}')
    # state = next_state
    # logger.debug(f'Sim {sim+1}, State: {state}, Reward: {reward}, Done: {done}, Winloss reward: {reward_detail[0]}, Blue dead reward: {reward_detail[1]}, Blue evacuated reward: {reward_detail[2]}, Red dead reward: {reward_detail[3]}')


# 初始化环境
env = RedBlueBattleEnv()

deployment_points_num = 40
NearTurret_num = 6
FarTurret_num = 6

angle_start = 6
Near_range = 5
Far_range = 6
action_dim = spaces.MultiDiscrete(
    [deployment_points_num, angle_start, Near_range] * NearTurret_num
    + [deployment_points_num, angle_start, Far_range] * FarTurret_num
)
state_dim = 9 * env.num_grids

gamma = 0.99  # 折扣因子
K_epochs = 4  # PPO的更新次数
clip_ratio = 0.2  # PPO的epsilon裁剪

# 初始化 PPO
# model = PPO(state_dim, action_dim, gamma, K_epochs)
model = PPO(
    map_channels=1,
    enemy_vector_dim=294 + 20,
    device=device,
    action_space=action_dim,
    gamma=gamma,
    K_epochs=K_epochs,
    is_train=False,
)
# 加载模型参数
save_path = 'logs/20240815_113500_experiment'
model.load(
    os.path.join(save_path, 'ppo_model_final_actor.pth'),
    os.path.join(save_path, 'ppo_model_final_critic.pth'),
)

style0_cnt = test(0, 1000, env)
# print(f"Style 0 success: {style0_cnt}")
style1_cnt = test(1, 1000, env)
# print(f"Style 1 success: {style1_cnt}")
style2_cnt = test(2, 1000, env)
# print(f"Style 2 success: {style2_cnt}")
style3_cnt = test(3, 1000, env)
# print(f"Style 3 success: {style3_cnt}")

print(
    f'''
      20240813_130108_experiment
    风格0：平均胜率{style0_cnt/10}%（千轮）
    风格1：平均胜率{style1_cnt/10}%（千轮）
    风格2：平均胜率{style2_cnt/10}%（千轮）
    混合：平均胜率{style3_cnt/10}%（千轮）

      '''
)
# # 推演循环
# num_simulations = 100  # 设定推演次数
# rewards_per_episode = []
# win_rates = []
# action_list = []

# 随机风格

# 集中风格

# 固定路径

# 混合风格


# import matplotlib.pyplot as plt

# # 绘制推演结果（可视化）

# import matplotlib.pyplot as plt
# # 计算每1个episode的平均胜率
# window_size = 10
# averaged_win_rates = [np.mean(win_rates[i:i+window_size]) for i in range(0, len(win_rates), window_size)]
# averaged_rewards = [np.mean(rewards_per_episode[i:i+window_size]) for i in range(0, len(rewards_per_episode), window_size)]
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(range(len(averaged_win_rates)), averaged_win_rates, label='Average Win Rate')
# plt.title(f'Average Win Rate per {window_size} Sims')
# plt.xlabel(f'Sim (in {window_size}s)')
# plt.ylabel('Average Win Rate')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(range(len(averaged_rewards)), averaged_rewards, color='red', label='Average Reward')
# plt.title(f'Average Reward per {window_size} Sims')
# plt.xlabel(f'Sim (in {window_size}s)')
# plt.ylabel('Average Reward')
# plt.grid(True)

# plt.tight_layout()
# plt.show()
