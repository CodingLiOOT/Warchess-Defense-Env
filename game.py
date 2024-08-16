import random

import gym
import numpy as np
from gym import spaces

import utils
from warchess import WarChessGame


class RedBlueBattleEnv(gym.Env):

    def __init__(self):
        super(RedBlueBattleEnv, self).__init__()
        self.done = False
        self.steps = 0
        self.simulator = WarChessGame(is_render=False, log_level='error')

        # self.simulator = WarChessGame(is_render=True, log_level='error') #info

        self.grid_width = self.simulator.get_map_width()
        self.grid_height = self.simulator.get_map_height()
        self.deployment_points = [utils.xy2idx(x, y) for x, y in self.simulator.deployment_points]
        self.num_grids = self.grid_width * self.grid_height

        # 格子坐标
        self.coords = np.array([(x, y) for x in range(self.grid_width) for y in range(self.grid_height)])

        # 定义每个属性的最小值和最大值
        self.low = np.zeros((self.num_grids, 9), dtype=np.int32)
        self.high = np.zeros((self.num_grids, 9), dtype=np.int32)

        # 地形类型 [0, 4] 0:街道, 1：建筑, 3：可部署点, 4：出生点, 4：撤离点
        self.high[:, 0] = 4

        # flak数量,0和1
        self.high[:, 1] = 1

        # far_turret数量,0和1
        self.high[:, 2] = 1

        # near_turret数量,0和1
        self.high[:, 3] = 1

        # soldier数量,0到30
        self.high[:, 4] = self.simulator.get_config_value('num_soldier')

        # drone数量,0到30
        self.high[:, 5] = self.simulator.get_config_value('num_drone')

        # 红方生命值 [0, 100]
        self.high[:, 6] = self.simulator.get_max_health()

        # 红方朝向 [0, 9], 0代表没有红方算子,9代表全方位朝向
        self.high[:, 7] = 9

        # 蓝方死亡单位数量
        self.high[:, 8] = self.simulator.blue_team_unit_num

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.int32)

        # 动作空间：炮塔类型（0到2），炮塔放置的坐标编号 (x, y)，以及朝向（0到8);
        red_team_unit_num = self.simulator.red_team_unit_num
        deployment_points_num = len(self.deployment_points)
        # self.action_space = spaces.MultiDiscrete([deployment_points_num, 8] *
        #                                          red_team_unit_num)
        self.action_space = spaces.MultiDiscrete([deployment_points_num] * red_team_unit_num)

        self.reset()

    def reset(self, deploy_type=0):
        self.done = False
        # 模拟器重启
        self.simulator.reset()
        # 蓝方部署
        self.simulator.blue_team_deployment(deploy_type)
        # 初始化状态：所有格子为空
        self.state = self.simulator.get_state()
        # print(f'state.shape:{self.state.shape}')
        # self.state = self.state.reshape(1, -1)

        return self.state

    def step(self, action):
        if self.done:
            raise RuntimeError("Environment is done. Reset it before calling step().")

        # 红方部署
        # num_flak = self.simulator.num_flak            #需要防空塔的时候加回来
        num_far_turret = self.simulator.num_far_turret
        num_near_turrer = self.simulator.num_near_turrer
        # print(action)
        action = np.array(action).reshape(-1, 3)

        red_team = []
        # for idx in range(num_flak):
        #     pos, direction = action[idx]
        #     pos = utils.idx2xy(self.deployment_points[pos])
        #     red_team.append([f'flak_{idx}', pos, direction + 1])
        # for idx in range(num_far_turret):
        #     pos, direction = action[num_flak + idx]
        #     pos = utils.idx2xy(self.deployment_points[pos])
        #     red_team.append([f'far_turret_{idx}', pos, direction + 1])
        # for idx in range(num_near_turrer):
        #     pos, direction = action[num_flak + num_far_turret + idx]
        #     pos = utils.idx2xy(self.deployment_points[pos])
        #     red_team.append([f'near_turret_{idx}', pos, direction + 1])

        # for idx in range(num_flak):
        #     pos = action[idx]
        #     pos = utils.idx2xy(self.deployment_points[pos[0]])
        #     red_team.append([f'flak_{idx}', pos])
        # for idx in range(num_far_turret):
        #     pos = action[num_flak + idx]
        #     pos = utils.idx2xy(self.deployment_points[pos[0]])
        #     red_team.append([f'far_turret_{idx}', pos])
        # for idx in range(num_near_turrer):
        #     pos = action[num_flak + num_far_turret + idx]
        #     pos = utils.idx2xy(self.deployment_points[pos[0]])
        #     red_team.append([f'near_turret_{idx}', pos])

        # # 解析和处理红方防空炮
        # for idx in range(num_flak):
        #     pos, angle_start, angle_range= action[idx]
        #     angle_start = int(angle_start)
        #     angle_end = int(angle_range) + angle_start
        #     angle = [0, 0]
        #     if angle_start < angle_end:
        #         angle[0] = angle_start
        #         angle[1] = angle_end
        #     else:
        #         angle[0] = angle_end
        #         angle[1] = angle_start

        #     pos = utils.idx2xy(self.deployment_points[int(pos)])
        #     red_team.append([f'flak_{idx}', pos, angle])

        # 解析和处理红方远程炮塔
        # start_idx = num_flak
        start_idx = 0
        for idx in range(num_far_turret):
            pos, angle_start, angle_range = action[start_idx + idx]
            angle_start = (int(angle_start) + 1) * 60
            angle_end = (int(angle_range) + 1) * 60 + angle_start

            # 标准化角度范围
            angle_start = angle_start % 360
            if angle_end >= 720:
                angle_end = angle_end - 360

            angle = [0, 0]
            angle[0] = angle_start
            angle[1] = angle_end

            pos = utils.idx2xy(self.deployment_points[int(pos)])
            red_team.append([f'far_turret_{idx}', pos, angle])

        # # 遍历 self.deployment_points 的所有值并保存到一个变量中
        # deployment_points_positions = [utils.idx2xy(point) for point in self.deployment_points]

        # # 打印保存的变量
        # print(f"部署点坐标: {deployment_points_positions}")

        # 解析和处理红方近程炮塔
        start_idx += num_far_turret
        for idx in range(num_near_turrer):
            pos, angle_start, angle_range = action[start_idx + idx]
            angle_start = (int(angle_start) + 1) * 60
            angle_end = (int(angle_range) + 1) * 60 + angle_start

            # 标准化角度范围
            angle_start = angle_start % 360
            if angle_end >= 720:
                angle_end = angle_end - 360

            angle = [0, 0]
            angle[0] = angle_start
            angle[1] = angle_end

            pos = utils.idx2xy(self.deployment_points[int(pos)])
            red_team.append([f'near_turret_{idx}', pos, angle])

        # print(red_team)
        self.simulator.red_team_deployment(red_team)

        # # 在这里进行推演，计算奖励
        # if np.unique(action[:,
        #                     0]).shape[0] != self.simulator.red_team_unit_num:
        #     self.state = self.simulator.get_state()
        #     reward = -500
        # else:
        #     result, blue_dead, blue_evacuated, blue_team_sum_distance, red_dead = self.simulator.simulate_battle()
        #     self.state = self.simulator.get_state()
        #     if result == 1:
        #         reward = 100
        #     elif result == 2:
        #         reward = -100
        #     elif result == 3:
        #         reward = -200
        #     else:
        #         reward = 0

        # profiler = cProfile.Profile()
        # profiler.enable()

        result, blue_dead, blue_evacuated, blue_team_sum_distance, red_dead = self.simulator.simulate_battle()

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumulative')
        # stats.print_stats()

        reward = 0
        dead_weight = 1.0
        evacuated_weight = -1.0
        distance_weight = -0.1
        red_dead_weight = -1.0

        # 根据游戏结果调整奖励或惩罚
        if result == 0:
            # 游戏超时，根据情况给予小的惩罚或奖励
            reward += 10
        elif result == 1:
            # 红方胜利，给予大奖励
            reward += 100
        elif result == 2:
            # 蓝方全部撤离，给予惩罚
            reward += -100
        elif result == 3:
            # 红方全部被摧毁，给予大惩罚
            reward += -100
        reward_detail = []
        reward_detail.append(reward)
        reward_detail.append(blue_dead * dead_weight)
        reward_detail.append(blue_evacuated * evacuated_weight)
        reward_detail.append(red_dead * red_dead_weight)

        # 根据蓝方死亡数量、撤离数量、距离总和和红方死亡数量计算额外奖励或惩罚
        # reward += (blue_dead * dead_weight +
        #            blue_evacuated * evacuated_weight +
        #            blue_team_sum_distance * distance_weight +
        #            red_dead * red_dead_weight)
        # reward += (blue_dead * dead_weight +
        #            blue_evacuated * evacuated_weight +
        #            red_dead * red_dead_weight)
        reward += 10 * blue_evacuated
        self.done = True
        self.steps += 1

        return self.state, reward, self.done, result, reward_detail

    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    env = RedBlueBattleEnv()
    state_len = []
    for i in range(100):
        state = env.reset(random.randint(0, 2))
        state_len.append(state['enemy_triples'].shape[0])
    print(state_len)
