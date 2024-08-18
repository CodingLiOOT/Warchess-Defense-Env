import argparse
import json
import os
import random
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
    # return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")




#def test(model, env, device):
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

    success_cnt0 = test_one_style(0, 100, env, model)
    success_cnt1 = test_one_style(1, 100, env, model)
    success_cnt2 = test_one_style(2, 100, env, model)
    success_cnt3 = test_one_style(3, 100, env, model)
    print(
        f'''
    风格0：平均胜率{success_cnt0/10}%（千轮）
    风格1：平均胜率{success_cnt1/10}%（千轮）
    风格2：平均胜率{success_cnt2/10}%（千轮）
    混合：平均胜率{success_cnt3/10}%（千轮）
      '''
    )
def test(model, env, device, save_dir):
    def test_one_style(style, num_simulation, env, model, save_dir):
        import random

        results = []
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

                # 在每一步记录相关信息
                step_info = {
                    "reward": reward,
                    "result": result,
                    "blue_evacuated": reward_detail[2],  # 撤离的蓝方数量
                    "red_dead": reward_detail[3],        # 死亡的红方数量
                    "blue_dead": reward_detail[1],       # 死亡的蓝方数量
                    "steps": env.steps                   # 当前步数
                }
                results.append(step_info)
                print(f"style:{style},episode:{i},result:{result}")
                state = next_state

                if result == 1:
                    succes_cnt += 1

            # results.append(episode_info)
        
        # 将结果保存到JSON文件
        with open(os.path.join(save_dir, f'test_results_style_{style}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return succes_cnt

    success_cnt0 = test_one_style(0, 1000, env, model, save_dir)
    success_cnt1 = test_one_style(1, 1000, env, model, save_dir)
    success_cnt2 = test_one_style(2, 1000, env, model, save_dir)
    success_cnt3 = test_one_style(3, 1000, env, model, save_dir)
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

    
    if args.mode == "test":
        if args.save_dir is None:
            raise ValueError("save_dir must be provided in test mode")
        model.actor = torch.load(
            os.path.join(args.save_dir, "ppo_model_final_actor.pth"), map_location=device
        )
        model.critic = torch.load(
            os.path.join(args.save_dir, "ppo_model_final_critic.pth"), map_location=device
        )
        test(model, env, device,args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", help="train or test")
    parser.add_argument("--save_dir", type=str, default="model", help="directory to save logs and models")
    args = parser.parse_args()
    main(args)
