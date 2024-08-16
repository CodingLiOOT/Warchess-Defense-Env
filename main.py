import argparse
import os
import time

import numpy as np
import torch
import torch.backends
from gym import spaces

from game import RedBlueBattleEnv  # 确保 game.py 文件中的环境可以被正确导入
from PPO import PPO  # 确保 PPO.py 文件中的类可以被正确导入
from tools.logging_utils import (
    close_writer,
    get_logger,
    init_logger,
    init_writer,
    log_scalar,
)


def select_device():
    """选择适合的装备：MPS、CUDA 或 CPU
    Returns:
        device (torch.device): 选择的装备
    """
    return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train(model, env, save_dir, is_render=False):
    # 初始化tensorboard和logger
    log_dir = os.path.join(save_dir, 'log')
    init_writer(log_dir)
    init_logger(log_dir)
    logger = get_logger('train')
    # 训练
    num_episodes = 12800
    # 滑动平均值列表
    rewards_per_episode = []
    win_rates = []

    for episode in range(num_episodes):
        state = env.reset(2)
        total_reward = 0
        done = False
        while not done:
            action, buffer_tuple = model.select_action(state)
            next_state, reward, done, result, reward_detail = env.step(action)
            total_reward += reward
            split_reward = reward / 12.0
            for i in range(12):
                single_state, single_action, single_log_probs, single_reward, single_done = buffer_tuple[i]
                single_reward = split_reward
                if i == 11:
                    single_done = True
                model.buffer.add(single_state, single_action, single_log_probs, single_reward, single_done)
            logger.info(f'Episode{episode+1},Reward:{reward},Result:{result}')
            # logger.debug(
            #     f'Episode {episode+1}, Reward: {reward}, Done: {done}, Winloss reward: {reward_detail[0]}, Blue dead reward: {reward_detail[1]}, Blue evacuated reward: {reward_detail[2]}, Red dead reward: {reward_detail[3]}'
            # )
            state = next_state

        rewards_per_episode.append(total_reward)
        win_rates.append(1 if result == 1 else 0)  # 胜利时记录1，否则记录0

        if (episode + 1) % 10 == 0:
            window_size = 10
            averaged_win_rate = np.mean(win_rates[-window_size:])
            averaged_reward = np.mean(rewards_per_episode[-window_size:])
            log_scalar('Average Win Rate', averaged_win_rate, episode + 1)
            log_scalar('Average Reward', averaged_reward, episode + 1)

        if (episode + 1) % 16 == 0:
            model.update()  # 更新模型
            # print(f'Episode {episode + 1}: Total Reward = {total_reward}')
            logger.info(f'Progress Update - Episode {episode+1}')
        if (episode + 1) % 1024 == 0:
            model.save(save_dir, 'ppo_model_' + str(episode + 1))
            logger.info(f'Model saved after {episode+1} episodes')
    close_writer()
    model.save(save_dir, 'ppo_model_final')
    logger.info('Model saved after training')


def test(model, env, device):
    def test_one_style(style, num_simulation, env, model):
        import random

        succes_cnt = 0
        for i in range(num_simulation):
            if style == 3:
                state = env.reset(random.randint(0, 2))
            else:
                state = env.reset(style)
            done = False
            while not done:
                map_input = state['map']
                enemy_vector = state['enemy_triples']
                action, log_probs = model.select_action(state)
                next_state, reward, done, result, reward_detail = env.step(action)
                if result == 1:
                    print(f'风格{style}：第{i}轮胜利')
                    succes_cnt += 1
        return succes_cnt

    success_cnt0 = test_one_style(0, 1000, env, model)
    success_cnt1 = test_one_style(1, 1000, env, model)
    success_cnt2 = test_one_style(2, 1000, env, model)
    success_cnt3 = test_one_style(3, 1000, env, model)
    print(
        f'''
    风格0：平均胜率{success_cnt0/10}%（千轮）
    风格1：平均胜率{success_cnt1/10}%（千轮）
    风格2：平均胜率{success_cnt2/10}%（千轮）
    混合：平均胜率{success_cnt3/10}%（千轮）
      '''
    )


def main(args):
    # 选择设备
    device = select_device()

    # 初始化环境
    env = RedBlueBattleEnv()

    # 初始化PPO
    model = PPO(
        map_channels=1,
        enemy_vector_dim=3 * 78 + 40 * 3,
        device=device,
        action_space=spaces.MultiDiscrete([40, 6, 6]),
        gamma=0.99,
        K_epochs=4,
        batch_size=16 * 12,
    )

    if args.mode == "train":
        save_dir = (
            f"logs/{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_experiment"
            if args.save_dir is None
            else args.save_dir
        )
        os.makedirs(save_dir, exist_ok=True)
        train(model, env, save_dir, device)
    elif args.mode == "test":
        if args.save_dir is None:
            raise ValueError("save_dir must be provided in test mode")
        model.actor = torch.load(
            os.path.join(args.save_dir, "ppo_model_final_actor.pth"), map_location=device
        )
        model.critic = torch.load(
            os.path.join(args.save_dir, "ppo_model_final_critic.pth"), map_location=device
        )
        test(model, env, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--save_dir", type=str, default=None, help="directory to save logs and models")
    args = parser.parse_args()
    main(args)
