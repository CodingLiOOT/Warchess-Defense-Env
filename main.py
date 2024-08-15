import os
import time
import torch
import argparse
import random
import numpy as np
import torch.backends
from PPO import PPO  # 确保 PPO.py 文件中的类可以被正确导入
from game import RedBlueBattleEnv  # 确保 game.py 文件中的环境可以被正确导入
from gym import spaces
from tools.logging_utils import init_logger, get_logger, init_writer, log_scalar, close_writer


def select_device():
    """选择适合的装备：MPS、CUDA 或 CPU
    Returns:
        device (torch.device): 选择的装备
    """
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
            map_input = state['map']
            enemy_vector = state['enemy_triples']
            action, log_probs = model.select_action(map_input, enemy_vector)
            next_state, reward, done, result, reward_detail = env.step(action)
            total_reward += reward
            model.buffer.add(state, action, log_probs, reward, done)
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

        if (episode + 1) % 64 == 0:
            model.update()  # 更新模型
            # print(f'Episode {episode + 1}: Total Reward = {total_reward}')
            logger.info(f'Progress Update - Episode {episode+1}')
        if (episode + 1) % 1000 == 0:
            model.save(save_dir, 'ppo_model' + str(episode + 1))
            logger.info(f'Model saved after {episode+1} episodes')


def test(model, env, device):
    pass


def main(args):
    # 选择设备
    device = select_device()

    # 初始化环境
    env = RedBlueBattleEnv()

    # 初始化PPO
    model = PPO(
        map_channels=1,
        enemy_vector_dim=294 + 20,
        device=device,
        action_space=spaces.MultiDiscrete([40, 6, 5] * 6 + [40, 6, 6] * 6),
        gamma=0.99,
        K_epochs=4,
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
        if save_dir is None:
            raise ValueError("save_dir must be provided in test mode")
        model.load(
            os.path.join(save_dir, "model.pth"),
            os.path.join(save_dir, "optimizer.pth"),
        )
        test(model, env, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--save_dir", type=str, default=None, help="directory to save logs and models")
    args = parser.parse_args()
    main(args)
