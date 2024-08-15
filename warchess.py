import copy
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import pygame as pg

import utils
from stone import *


class Map:

    def __init__(self, save_path='', is_render=True):
        self.map = []
        self.width = utils.map_width
        self.height = utils.map_height
        self.lenght_map = self.width * self.height

        pg.font.init()
        self.font = pg.font.Font(None, 10)

        # key point
        self.deployment_points = []
        self.entrances_points = []
        self.exit_points = []
        self.waypoints = []

        # screen
        self.save_path = save_path
        self.rect = None
        self.screen = None
        if is_render:
            self.screen = pg.display.set_mode(
                (self.width * utils.x_len, self.height * utils.y_len), flags=0)
            pg.display.set_caption('Defense Simulation')
            self.rect = self.screen.get_rect()
            self.rect.x = 0
            self.rect.y = 0

        # 通视文件路径
        self.visibility_map_file = os.path.join('resource', 'visibility_map.json')  
        # 通视地图
        self.visibility_map = None
        # 通视地图计数器
        self.visibility_map_clicker = 0
        # 可部署点坐标
        self.deployment_points_positions = [(74, 60), (74, 62), (74, 64), (74, 73), (74, 75), (74, 77), (76, 60), (76, 64), (76, 73), (76, 77), (78, 60), (78, 62), (78, 64), (78, 73), (78, 75), (78, 77), (83, 68), (83, 70), (83, 72), (85, 68), (85, 72), (87, 68), (87, 70), (87, 72), (96, 60), (96, 62), (96, 64), (96, 73), (96, 75), (96, 77), (98, 60), (98, 64), (98, 73), (98, 77), (100, 60), (100, 62), (100, 64), (100, 73), (100, 75), (100, 77)]

        self.range_clicker = 0
    # map
    # 0:街道, 1:建筑, 2:建筑(可部署), 3:部署位置, 4:出生点, 5:撤离点, 6:路径点
    def setup_map(self):
        self.map = []
        for n in range(self.lenght_map):
            x, y = utils.idx2xy(n)
            m = utils.map[y][x]
            # print(f"处理索引: {n}, 坐标: ({x}, {y})")
            self.map.append(m)
            if m == 3:
                self.deployment_points.append((x, y))
            elif m == 4:
                self.entrances_points.append((x, y))
                self.waypoints.append((x, y))
            elif m == 5:
                self.exit_points.append((x, y))
                self.waypoints.append((x, y))
            elif m == 6:
                self.waypoints.append((x, y))

        self.deployment_points.sort()
        self.exit_points.sort()
        self.waypoints.sort()
        # self.get_visibility_map_from_json()
        if self.visibility_map_clicker == 0:
            self.visibility_map_clicker += 1
            self.get_visibility_map_from_json()
    def get_2d_map(self):
        """
        辅助方法：将一维地图数组转换为二维数组
        """
        return np.array(self.map).reshape((self.height, self.width))
    def get_waypoints(self):
        return self.waypoints, self.entrances_points, self.exit_points

    def get_deployment_points(self):
        return self.deployment_points

    # visualization
    def drawback_ground(self, surface):
        color = None
        for n in range(self.lenght_map):
            x = n % self.width
            y = n // self.width
            if self.map[n] == 0:
                color = (247, 238, 214)
            elif self.map[n] == 1:
                color = (166, 62, 0)
                # color = (0, 60, 201)
            elif self.map[n] == 2:
                color = (45, 45, 100)
            elif self.map[n] == 3:
                color = (45, 45, 100)
            elif self.map[n] == 4:
                color = (0, 255, 0)
            elif self.map[n] == 5:
                color = (0, 0, 255)
            elif self.map[n] == 6:
                color = (247, 238, 214)
            base_x, base_y = x * utils.x_len, y * utils.y_len
            pg.draw.rect(surface, color,
                         (base_x, base_y, utils.x_len, utils.y_len))
        surface.blit(surface, self.rect)

    def draw_line(self):
        for n in range(self.lenght_map):
            x = n % self.width
            y = n // self.width
            base_x, base_y = x * utils.x_len, y * utils.y_len
            pg.draw.rect(self.screen, (0, 0, 0),
                         (base_x, base_y, utils.x_len, utils.y_len), 1)

    def drawback_ground_preview(self, surface):
        color = None
        for n in range(self.lenght_map):
            x = n % self.width
            y = n // self.width
            if self.map[n] == 0:
                color = (247, 238, 214)
            elif self.map[n] == 1:
                color = (166, 62, 0)
            elif self.map[n] == 2:
                color = (45, 45, 100)
            elif self.map[n] == 3:
                color = (255, 0, 0)
            elif self.map[n] == 4:
                color = (0, 255, 0)
            elif self.map[n] == 5:
                color = (0, 0, 255)
            elif self.map[n] == 6:
                color = (0, 255, 255)
            base_x, base_y = x * utils.x_len, y * utils.y_len
            pg.draw.rect(surface, color,
                         (base_x, base_y, utils.x_len, utils.y_len))
        surface.blit(surface, self.rect)

    def drawback_route(self, surface, route):
        color = None
        for n in range(self.lenght_map):
            x = n % self.width
            y = n // self.width
            if self.map[n] == 0:
                color = (247, 238, 214)
            elif self.map[n] == 1:
                color = (166, 62, 0)
            elif self.map[n] == 2:
                color = (45, 45, 100)
            elif self.map[n] == 3:
                color = (45, 45, 100)
            elif self.map[n] == 4:
                color = (0, 255, 0)
            elif self.map[n] == 5:
                color = (0, 0, 255)
            elif self.map[n] == 6:
                color = (247, 238, 214)
            if (x, y) in route:
                color = (200, 5, 5)
            base_x, base_y = x * utils.x_len, y * utils.y_len
            pg.draw.rect(surface, color,
                         (base_x, base_y, utils.x_len, utils.y_len))
        surface.blit(surface, self.rect)
        


    # def to_pygame_angle(degrees):
    #     """
    #     将角度从正上方开始顺时针增大的形式转换为 Pygame 使用的逆时针增大的形式。
        
    #     :param degrees: 顺时针增大的角度
    #     :return: Pygame 使用的逆时针角度
    #     """
    #     return (360 - degrees) % 360
    
    # def draw_arc(screen, color, center, radius, start_angle, end_angle, width=1):
    #     """
    #     在屏幕上绘制一个圆弧，按照正上方为 0 度，沿顺时针方向增大的要求。
        
    #     :param screen: Pygame 的屏幕对象
    #     :param color: 圆弧的颜色
    #     :param center: 圆心坐标 (x, y)
    #     :param radius: 圆弧的半径
    #     :param start_angle: 圆弧的起始角度（以度为单位，从正上方开始，顺时针增加）
    #     :param end_angle: 圆弧的终止角度（以度为单位，从正上方开始，顺时针增加）
    #     :param width: 圆弧的宽度
    #     """
    #     # 将角度转换为 Pygame 逆时针方向角度
    #     start_angle_pygame = math.radians(to_pygame_angle(start_angle))
    #     end_angle_pygame = math.radians(to_pygame_angle(end_angle))
        
    #     # Pygame 中的角度范围是从 0 到 2π 弧度
    #     # 转换成 Pygame 需要的格式
    #     start_angle_pygame = math.radians(360 - start_angle)
    #     end_angle_pygame = math.radians(360 - end_angle)
        
    #     # 绘制圆弧
    #     pg.draw.arc(screen, color, pg.Rect(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius),
    #                 start_angle_pygame, end_angle_pygame, width)

    # def draw_range_on_map(self, screen, stone):
    #     # 创建一个与屏幕同尺寸的透明表面
    #     overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
        
    #     # 遍历地图上的每个位置
    #     for n in range(self.lenght_map):
    #         x = n % self.width
    #         y = n // self.width
            
            
    #         if stone.is_enemy_in_range(x, y):
    #             # 基础颜色：浅蓝色
    #             base_color = pg.Color(173, 216, 230)
                
    #             # 获取当前点的颜色
    #             current_color = overlay.get_at((x, y))
            
    #             # 计算新的颜色
    #             if current_color.a > 0 and (current_color.r, current_color.g, current_color.b) == (173, 216, 230):
    #                 # 如果当前颜色是浅蓝色，增加透明度以使颜色更深
    #                 new_alpha = min(255, current_color.a + 50)  # 增加透明度
    #                 new_color = pg.Color(base_color.r, base_color.g, base_color.b, new_alpha)
    #             else:
    #                 # 如果当前点没有颜色或颜色不是浅蓝色，使用基础颜色
    #                 new_color = base_color
                
    #             ex, ey = x * utils.x_len, y * utils.y_len
    #             # 设置颜色到透明表面
    #             overlay.set_at((ex, ey), new_color)

    #             pg.draw.rect(screen, new_color,
    #                      (ex, ey, utils.x_len, utils.y_len))
    #     # 将透明表面绘制到屏幕上
        
    #     screen.blit(screen, self.rect)

    def draw_range_on_map(self, screen, stone):
        overlay = pg.Surface((self.width * utils.x_len, self.height * utils.y_len), pg.SRCALPHA)
        overlay.set_alpha(50)  # 设置整个图层的透明度，这里设置为128（0-255范围）

        for n in range(self.lenght_map):
            x = n % self.width
            y = n // self.width
            
            if stone.is_enemy_in_range(x, y):
                base_color = pg.Color(63, 120, 252)
                # base_color = pg.Color(63, 120, 252)
                current_color = overlay.get_at((x, y))
                
                if current_color.a > 0 and (current_color.r, current_color.g, current_color.b) == (63, 120, 252):
                    new_alpha = min(255, current_color.a + 50)
                    new_color = pg.Color(base_color.r, base_color.g, base_color.b, new_alpha)
                else:
                    new_color = base_color
                
                ex, ey = x * utils.x_len, y * utils.y_len
                pg.draw.rect(overlay, new_color,
                            (ex, ey, utils.x_len, utils.y_len))
        
        screen.blit(overlay, (0, 0))
    def drew_unit(self, unit_list):
        unit_type_list = list(set(x.get_stone_type() for x in unit_list))
        unit_color = {}
        unit_image = {}
        unit_count = np.zeros((self.lenght_map, len(unit_type_list)),
                              dtype=int)
        unit_directions = np.zeros((self.lenght_map, len(unit_type_list)),
                                   dtype=int)

        for unit in unit_list:
            unit_type = unit_type_list.index(unit.get_stone_type())
            unit_pos = utils.xy2idx(*unit.get_pos())
            if unit_type not in unit_color:
                unit_color[unit_type] = unit.get_color()
                unit_image[unit_type] = unit.get_image()
            if unit.get_states() not in [0, 3]:  # 不显示已死亡与已撤退单位
                unit_count[unit_pos, unit_type] += 1
                unit_directions[unit_pos, unit_type] = unit.get_direction()[0]

        for pos in range(self.lenght_map):
            if unit_count[pos].sum() > 0:
                color = []
                for i in range(len(unit_type_list)):
                    count = unit_count[pos][i]
                    count = -1 if count > len(unit_color[i]) else count - 1
                    color.append((unit_color[i][count], unit_count[pos][i]))
                color = utils.mix_colors(color)
                x, y = utils.idx2xy(pos)
                base_x, base_y = x * utils.x_len, y * utils.y_len
                rect_surface = pg.Surface((utils.x_len, utils.y_len),
                                          pg.SRCALPHA)
                rect_surface.fill(color)
                self.screen.blit(rect_surface, (base_x, base_y))

            for i in range(len(unit_type_list)):
                if unit_count[pos][i] > 0 and unit_image[i] is not None:
                    image = unit_image[i]
                    direction = unit_directions[pos][i]
                    if direction > -1:
                        angle = (direction - 2) * 45
                        image = pg.transform.rotate(image, angle)
                    rect = image.get_rect()
                    image = pg.transform.scale(image,
                                               (utils.image_x, utils.image_y))
                    x, y = utils.idx2xy(pos)
                    base_x, base_y = x * utils.x_len, y * utils.y_len
                    rect.topleft = (base_x, base_y)
                    self.screen.blit(image, rect)
        # if self.range_clicker == 0:
        #     self.range_clicker += 1
        #     self.get_visibility_map_from_json()
        #     for stone in unit_list:
        #         if stone.name.startswith('near') and stone.get_states() != 0:
        #             self.draw_range_on_map(self.screen, stone)

        #         if stone.name.startswith('far'):
        #             self.draw_range_on_map(self.screen, stone)
        for stone in unit_list:
                if stone.name.startswith('near') and stone.get_states() != 0:
                    self.draw_range_on_map(self.screen, stone)

                if stone.name.startswith('far') and stone.get_states() != 0:
                    self.draw_range_on_map(self.screen, stone)

    def get_visibility_map_from_json(self):
        """
        从本地的 JSON 文件中读取通视地图并赋值给 self.visibility_map
        """
        try:
            with open(self.visibility_map_file, 'r') as f:
                visibility_map_list = json.load(f)
            self.visibility_map = np.array(visibility_map_list)
        except FileNotFoundError:
            print(f"文件 {self.visibility_map_file} 未找到，请先生成并保存通视地图。")
        except json.JSONDecodeError:
            print(f"文件 {self.visibility_map_file} 不是有效的 JSON 文件。")

    def save_visibility_map_to_json(self, visibility_map):
        """
        将通视地图保存到本地的 JSON 文件中
        """
        visibility_map_list = visibility_map.tolist()  # 转换为列表格式
        with open(self.visibility_map_file, 'w') as f:
            json.dump(visibility_map_list, f)

    def generate_visibility_map(self, deployment_points_positions):
        """
        生成通视地图，只存储 deployment_points_positions 之间的通视情况
        """
        num_deployment_points = len(deployment_points_positions)
        num_map_points = self.lenght_map  # 总的地图点数

        # 创建通视矩阵，行表示 deployment points，列表示地图中的每个点
        visibility_map = np.zeros((num_deployment_points, num_map_points), dtype=bool)
        
        for i, (x1, y1) in enumerate(deployment_points_positions):
            for j in range(self.lenght_map):
                x2, y2 = utils.idx2xy(j)  # 将地图索引转换为坐标
                if self.check_visibility((x1, y1), (x2, y2)):
                    visibility_map[i, j] = True

        return visibility_map
        
    def is_visible(self, unit1_pos, unit2_pos):
        """
        判断两个单位之间是否有建筑阻隔
        :param unit1_pos: 单位1的位置 (x1, y1)
        :param unit2_pos: 单位2的位置 (x2, y2)
        :return: True if no building obstructs the line of sight, False otherwise
        """
        is_unit1_deployment_point = unit1_pos in self.deployment_points_positions
        is_unit2_deployment_point = unit2_pos in self.deployment_points_positions

        # idx1 = utils.xy2idx(*unit1_pos)
        # idx2 = utils.xy2idx(*unit2_pos)

        if is_unit1_deployment_point:
            idx1 = self.deployment_points_positions.index(unit1_pos)
            idx2 = utils.xy2idx(*unit2_pos)
            
            # print(self.visibility_map[idx1][idx2])
            return self.visibility_map[idx1][idx2]
            
        if is_unit2_deployment_point:
            idx2 = self.deployment_points_positions.index(unit2_pos)
            idx1 = utils.xy2idx(*unit1_pos)  # 将unit1_pos转换为地图索引
            # 检查并返回通视情况
            
            
            # print(self.visibility_map[idx2][idx1])
            return self.visibility_map[idx2][idx1]
        # # idx1 = unit1_pos
        # # idx2 = unit2_pos
        
        # # 检查是否存储了 unit1 到 unit2 的通视情况
        # if idx1 in self.visibility_map and idx2 in self.visibility_map[idx1]:
        #     return self.visibility_map[idx1][idx2]
        # elif idx2 in self.visibility_map and idx1 in self.visibility_map[idx2]:
        #     # 如果 unit2 到 unit1 的通视情况被存储，那么也可以返回该结果
        #     return self.visibility_map[idx2][idx1]
        # else:
        #     # 如果两者都没有直接存储，可能需要额外的逻辑来处理或者返回默认值
        #     # 这种情况下，你可能需要重新考虑如何生成 visibility_map，或者考虑在此处计算通视
        #     return False

    def check_visibility(self, unit1_pos, unit2_pos):
        """
        判断两个单位之间是否有建筑阻隔（内部使用）
        :param unit1_pos: 单位1的位置 (x1, y1)
        :param unit2_pos: 单位2的位置 (x2, y2)
        :return: True if no building obstructs the line of sight, False otherwise
        """
        x1, y1 = unit1_pos
        x2, y2 = unit2_pos
        map_2d = self.get_2d_map()

        # 使用 Bresenham's line algorithm 计算单位之间的直线
        line_points = self.bresenham_line(x1, y1, x2, y2)

        # 检查直线上的每一个点，是否有建筑阻隔
        for (x, y) in line_points:
            if map_2d[y][x] == 1:
                return False

        return True

    def bresenham_line(self, x1, y1, x2, y2):
        """
        使用 Bresenham's line algorithm 计算两点之间的直线
        :param x1: 起点 x 坐标
        :param y1: 起点 y 坐标
        :param x2: 终点 x 坐标
        :param y2: 终点 y 坐标
        :return: 两点之间的直线上的点的列表
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points

    def filter_visible_units(self, attacker, targets):
        """
        过滤掉那些与攻击者之间没有通视的目标
        :param attacker: 攻击者单位
        :param targets: 目标单位列表
        :return: 满足通视条件的目标单位列表
        """
        visible_targets = []
        attacker_pos = attacker.get_pos()
        # attacker_posx, attacker_posy = attacker_pos

        for target in targets:
            target_pos = target.get_pos()
            # target_posx, target_posy = target_pos

            dx = target_pos[0] - attacker_pos[0]
            dy = target_pos[1] - attacker_pos[1]
            enemy_distance = math.sqrt(dx * dx + dy * dy)

            # 如果敌人距离大于防御塔的最远攻击距离或小于最近攻击距离，则敌人在攻击范围外
            if (attacker.attack_range[0] <= enemy_distance <= attacker.attack_range[1]):
                # print(self.is_visible(attacker_pos, target_pos))
                if self.is_visible(attacker_pos, target_pos):
                    visible_targets.append(target)
    
        return visible_targets
    
    def save_screen(self, name=None):
        name = 'frame' if name is None else name
        if os.path.isdir(self.save_path):
            frame_filename = os.path.join(
                self.save_path, '{}_{}.png'.format(name, int(time.time())))
            pg.image.save(self.screen, frame_filename)

    def update(self, unit_list, step=None):
        self.drawback_ground(self.screen)
        self.drew_unit(unit_list)
        self.draw_line()
        self.save_screen(step)
        pg.display.flip()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

    def draw_route(self, route):
        self.drawback_route(self.screen, route)
        self.draw_line()
        pg.display.flip()

    def draw_game(self):
        self.drawback_ground_preview(self.screen)
        self.draw_line()
        self.save_screen(0)
        pg.display.flip()

    def quit(self):
        pg.quit()


class WarChessGame(object):

    def __init__(self,
                 config_path='config/config.json',
                 is_render=True,
                 log_level='error'):
        self.is_render = is_render
        self.map = None
        self.log_level = log_level

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.reset()

    def reset(self):
        self.step = 0
        self.red_team = []
        self.blue_team = []
        self.blue_team_preparation = []

        self.log_path = 'wargame_log/{}'.format(
            time.strftime("%y%m%d%H%M", time.localtime(time.time())))

        self.set_logger()
        self.set_map()

    @property
    def num_flak(self):
        return self.config['num_flak']

    @property
    def num_far_turret(self):
        return self.config['num_far_turret']

    @property
    def num_near_turrer(self):
        return self.config['num_near_turrer']

    @property
    def red_team_unit_num(self):
        return self.config['num_flak'] + self.config[
            'num_far_turret'] + self.config['num_near_turrer']

    @property
    def blue_team_unit_num(self):
        return self.config['num_soldier'] + self.config['num_drone']

    @property
    def deployment_points(self):
        if self.map is None:
            return []
        else:
            return [x for x in self.map.get_deployment_points()]

    def get_config_value(self, x=''):
        return self.config.get(x, None)

    def get_max_health(self):
        health = [v for k, v in self.config.items() if 'health' in k]
        return max(health)

    def update_terrain(self,terrain, enemy_positions, max_radius=5):
        """
        更新二维地形数组，扩展敌人覆盖范围。

        参数:
        terrain (2D np.array): 二维地形数组，值为0, 1, 2, 3, 4表示不同地形。
        enemy_positions (list of tuples): 敌人位置的列表，每个元素为(x, y, n)，其中x, y是二维数组下标，n是敌人数量。
        max_radius (int): 最大覆盖半径（范围）。

        返回:
        2D np.array: 更新后的二维地形数组。
        """
        # 计算敌人数量的最大值和最小值用于归一化
        max_enemies = max(enemy[2] for enemy in enemy_positions)
        
        for (x, y, n) in enemy_positions:
            # 归一化敌人数量并计算覆盖范围半径
            normalized_n = n / max_enemies  # 归一化到[0, 1]之间
            radius = int(normalized_n * max_radius)  # 映射到[0, max_radius]之间

            # 遍历覆盖范围内的所有格子
            for i in range(max(0, x - radius), min(terrain.shape[0], x + radius + 1)):
                for j in range(max(0, y - radius), min(terrain.shape[1], y + radius + 1)):
                    # 更新地形值为4
                    terrain[i, j] = 3

        return terrain

    def get_state(self):
        # 将地图转换为NumPy数组并进行值替换
        # map = np.array(self.map.map)
        # map[map == 2] = 1
        # map[map == 3] = 2
        # map[map == 4] = 3
        # map[map == 5] = 4
        # map[map == 6] = 0
        # print(map)
        # print(map.shape)
        # 调整目标大小为128x128
        # map = map.reshape(180, 135)

        with open('resource/grounding_path.json', 'r') as file:
            data = json.load(file)

        # 提取所有起始点，并将其转换为元组
        start_points = [tuple(route['start']) for route in data['routes']]

        # 初始化每个出发点的敌人数量
        enemy_counts = {start: 0 for start in start_points}

        for unit in self.blue_team + self.blue_team_preparation:
            pos = tuple(unit.get_pos())  # 获取单位的起始位置并转换为元组
            if pos in enemy_counts:
                enemy_counts[pos] += 1  # 更新对应出发点的敌人数量

        # 将字典转换为三元组列表
        enemy_triples = [(x, y, count) for (x, y), count in enemy_counts.items()]
        # print(f"Start point:{enemy_triples}")
        map = self.map.get_2d_map()
        map[map == 2] = 1
        map[map == 3] = 2
        map[map == 4] = 3
        map[map == 5] = 4
        map[map == 6] = 0
        self.update_terrain(map,enemy_triples,max_radius=8)
        map = np.expand_dims(map, axis=(0, 1)).transpose(0, 1, 3, 2)

        # 将enemy_triples转换为一维数组
        flat_enemies = np.array(enemy_triples).flatten()
        # 部署位置
        indices = np.where(map == 2)
        flattend_indices = np.array([index for pair in zip(indices[0],indices[1])for index in pair])
        features = np.concatenate([flat_enemies,flattend_indices])
        state = {
            "map": map,
            "enemy_triples": features,
        }
        return state

    def get_map_width(self):
        return self.map.width

    def get_map_height(self):
        return self.map.height

    def set_logger(self):

        if self.log_level == 'info':
            log_level = logging.INFO
        else:
            log_level = logging.ERROR

        os.makedirs(self.log_path, exist_ok=True)

        self.logger = logging.getLogger('warchess_game')
        if not self.logger.hasHandlers():         
            self.logger.setLevel(log_level)
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setLevel(log_level)
            self.file_handler = logging.FileHandler(
                os.path.join(self.log_path, 'result.log'))
            self.file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | Preparation | %(message)s')
            self.console_handler.setFormatter(formatter)
            self.file_handler.setFormatter(formatter)
            self.logger.handlers = [self.console_handler, self.file_handler]
    def set_map(self):
        self.map = Map(save_path=self.log_path, is_render=self.is_render)
        self.map.setup_map()

        if self.is_render:
            self.map.draw_game()

    # def blue_team_deployment(self, a =1):
    #     ground_routing, aerial_routing = utils.get_routing()
    #     # print(ground_routing)
    #     random.shuffle(ground_routing)
    #     random.shuffle(aerial_routing)
    #     assert self.blue_team_unit_num >= self.config[
    #         'minimum_unit_evacuated'], 'Insufficient blue team units'

    #     units_list = []
    #     # for idx in range(self.config['num_drone']):
    #     #     units_list.append(f'drone_{idx}')
    #     for idx in range(self.config['num_soldier']):
    #         units_list.append(f'soldier_{idx}')
    #     random.shuffle(units_list)

    #     for unit_name in units_list:
    #         if unit_name.startswith('drone'):
    #             (x, y), route = aerial_routing.pop()
    #             self.blue_team_preparation.append(
    #                 StoneDrone(unit_name,
    #                            x,
    #                            y,
    #                            route,
    #                            -1,
    #                            mobility=self.config['drone_mobility'],
    #                            health=self.config['drone_health'],
    #                            attack=self.config['drone_attack'],
    #                            attack_range=self.config['drone_attack_range'],
    #                            accuracy=self.config['drone_accuracy'],
    #                            ammunition=self.config['drone_ammunition']))
    #         elif unit_name.startswith('soldier'):
    #             (x, y), route = ground_routing.pop()
    #             print(type(route))
    #             print(f'route: {route}')
    #             self.blue_team_preparation.append(
    #                 StoneSoldier(
    #                     unit_name,
    #                     x,
    #                     y,
    #                     route,
    #                     -1,
    #                     mobility=self.config['soldier_mobility'],
    #                     health=self.config['soldier_health'],
    #                     attack=self.config['soldier_attack'],
    #                     attack_range=self.config['soldier_attack_range'],
    #                     accuracy=self.config['soldier_accuracy'],
    #                     ammunition=self.config['soldier_ammunition']))
    
    def blue_team_deployment(self, deploy_type=0 ):
        # 打开 JSON 文件并加载数据
        with open('resource/grounding_path.json', 'r') as file:
            data = json.load(file)

        # # 随机选择一个路由
        # selected_route = random.choice(data['routes'])

        # # 提取起点 (x, y)
        # start = selected_route['start']

        # # 随机选择路径中的一个数组
        # route = random.choice(selected_route['path'])

        # # 现在 start 和 route 包含所需的起点和路径数据
        # print("Start point:", start)
        # print("Selected route:", route)

        assert self.blue_team_unit_num >= self.config[
            'minimum_unit_evacuated'], 'Insufficient blue team units'

        units_list = []
        # for idx in range(self.config['num_drone']):
        #     units_list.append(f'drone_{idx}')
        for idx in range(self.config['num_soldier']):
            units_list.append(f'soldier_{idx}')
        random.shuffle(units_list)

        #随机出发点和路线
        if deploy_type == 0:
            for unit_name in units_list:
                if unit_name.startswith('soldier'):
                    # 随机选择一个路由
                    selected_route = random.choice(data['routes'])

                    # 提取起点 (x, y)
                    (x, y) = selected_route['start']

                    # 随机选择路径中的一个数组
                    route = copy.deepcopy(random.choice(selected_route['path']))
                    self.blue_team_preparation.append(
                        StoneSoldier(
                            unit_name,
                            x,
                            y,
                            route,
                            -1,
                            mobility=self.config['soldier_mobility'],
                            health=self.config['soldier_health'],
                            attack=self.config['soldier_attack'],
                            attack_range=self.config['soldier_attack_range'],
                            accuracy=self.config['soldier_accuracy'],
                            ammunition=self.config['soldier_ammunition']))
        # 同一个出发点不同路线            
        elif deploy_type == 1: 
            routes = np.array(data['routes'])
            routes_group = []
            # 边界条件
            conditions = {
                'left': lambda item: item['start'][0] < 10,
                'right': lambda item: item['start'][0] > 170,
                'top': lambda item: item['start'][1] < 10,
                'bottom': lambda item: item['start'][1] > 130
            }
            # 生成边界组
            for condition in conditions.values():
                routes_group.append([item for item in routes if condition(item)])
            # 中间组：不符合任何边界组条件的点,去掉中间组
            # routes_group.append([item for item in routes if not any(condition(item) for condition in conditions.values())])
            # 随机选择一个路由组
            selected_routes = random.choice(routes_group)
            
            for unit_name in units_list:
                if unit_name.startswith('soldier'):
                    selected_route=random.choice(selected_routes)
                    (x, y) = selected_route['start']
                    route = copy.deepcopy(random.choice(selected_route['path']))
                    self.blue_team_preparation.append(
                        StoneSoldier(
                            unit_name,
                            x,
                            y,
                            route,
                            -1,
                            mobility=self.config['soldier_mobility'],
                            health=self.config['soldier_health'],
                            attack=self.config['soldier_attack'],
                            attack_range=self.config['soldier_attack_range'],
                            accuracy=self.config['soldier_accuracy'],
                            ammunition=self.config['soldier_ammunition']))
        # 不同出发点，各出发点路线相同            
        elif deploy_type == 2:
            paths = []
            for route in data['routes']:
                paths.append({"start":route['start'],"route":random.choice(route['path'])})
            for unit_name in units_list:
                if unit_name.startswith('soldier'):
                     # 随机选择路径中的一个数组
                    path = random.choice(paths)
                    (x,y)=path['start']
                    route = copy.deepcopy(path['route'])
                    self.blue_team_preparation.append(
                        StoneSoldier(
                            unit_name,
                            x,
                            y,
                            route,
                            -1,
                            mobility=self.config['soldier_mobility'],
                            health=self.config['soldier_health'],
                            attack=self.config['soldier_attack'],
                            attack_range=self.config['soldier_attack_range'],
                            accuracy=self.config['soldier_accuracy'],
                            ammunition=self.config['soldier_ammunition']))      

    def red_team_deployment(self, units_list):
        """
        units_list -> [unit_name:str, posx:int, posy:int, directions:list]
        unit_name -> '{stone_type}_{idx}'
        directions -> [direction:int]
        """
        assert (self.red_team_unit_num) < len(
            self.deployment_points), 'Insufficient deployment points'

        # for unit in units_list:
        #     unit_name, (x, y), directions = unit
        #     if unit_name.startswith('flak'):
        #         self.logger.info(f'{unit_name} is deployed to ({x},{y}).')
        #     else:
        #         self.logger.info(
        #             f'{unit_name} is deployed to ({x},{y}), aim to {directions}.'
        #         )
        #     self.update_map()
        #     if unit_name.startswith('flak'):
        #         self.red_team.append(
        #             StoneFlak(name=unit_name,
        #                       posx=x,
        #                       posy=y,
        #                       reward=-1,
        #                       health=self.config['flak_health'],
        #                       attack=self.config['flak_attack'],
        #                       attack_range=self.config['flak_attack_range'],
        #                       accuracy=self.config['flak_accuracy'],
        #                       ammunition=self.config['flak_ammunition']))
        #     elif unit_name.startswith('near_turrer'):
        #         self.red_team.append(
        #             StoneNearTurret(
        #                 name=unit_name,
        #                 posx=x,
        #                 posy=y,
        #                 direction=[directions],
        #                 reward=-1,
        #                 health=self.config['near_turret_health'],
        #                 attack=self.config['near_turret_attack'],
        #                 angle=self.config['near_turret_angle'],
        #                 attack_range=self.config['near_turret_attack_range'],
        #                 accuracy=self.config['near_turret_accuracy'],
        #                 ammunition=self.config['near_turret_ammunition']))
        #     elif unit_name.startswith('far_turret'):
        #         self.red_team.append(
        #             StoneFarTurret(
        #                 name=unit_name,
        #                 posx=x,
        #                 posy=y,
        #                 direction=[directions],
        #                 reward=-1,
        #                 health=self.config['far_turret_health'],
        #                 attack=self.config['far_turret_attack'],
        #                 angle=self.config['far_turret_angle'],
        #                 attack_range=self.config['far_turret_attack_range'],
        #                 accuracy=self.config['far_turret_accuracy'],
        #                 ammunition=self.config['far_turret_ammunition']))

        temp_Flak_num = 0
        for unit in units_list:
            unit_name, (x, y), angle = unit
            if unit_name.startswith('flak'):
                self.logger.info(f'{unit_name} is deployed to ({x},{y}).')
            if unit_name.startswith('far_turret'):
                
                self.logger.info(f'{unit_name} is deployed to ({x},{y}).' + '\n' + f'angle from {angle[0]} to {angle[1]}')
            # else:
            #     # if self.deployment_points.index((x,y)) in [0,5,10,15,17]:
            #     #     directions = 8
            #     #     self.logger.info(
            #     #         f'{unit_name} is deployed to ({x},{y}), aim to {directions}.'
            #     #     )
            #     # if self.deployment_points.index((x,y)) in [1,6,11,16,18]:
            #     #     directions = 2
            #     #     self.logger.info(
            #     #         f'{unit_name} is deployed to ({x},{y}), aim to {directions}.'
            #     #     )
            #     # if self.deployment_points.index((x,y)) in [3,8,13,21,23]:
            #     #     directions = 6
            #     #     self.logger.info(
            #     #         f'{unit_name} is deployed to ({x},{y}), aim to {directions}.'
            #     #     )
            #     # if self.deployment_points.index((x,y)) in [4,9,14,22,24]:
            #     #     directions = 4
            #     #     self.logger.info(
            #     #         f'{unit_name} is deployed to ({x},{y}), aim to {directions}.'
            #     #     )

            #     # 索引到方向的映射字典
            #     # index_to_directions = {
            #     #     (0, 5, 10, 15, 17): 8,
            #     #     (1, 6, 11, 16, 18): 2,
            #     #     (3, 8, 13, 21, 23): 6,
            #     #     (4, 9, 14, 22, 24): 4
            #     #     }
            #     index_to_directions = {
            #         (0, 4, 8, 12, 16): 8,
            #         (1, 5, 9, 13, 17): 2,
            #         (2, 6, 10, 14, 18): 6,
            #         (3, 7, 11, 15, 19): 4
            #         }
                
            #     try:
            #         # 获取 (x, y) 在 self.deployment_points 中的索引
            #         index = self.deployment_points.index((x, y))
    
            #         # 遍历字典，确定 index 是否在某个键的元组中
            #         for key, directions in index_to_directions.items():
            #             if index in key:
            #                 self.logger.info(
            #                     f'{unit_name} is deployed to ({x},{y}), aim to {directions}.'
            #                 )
            #                 break
            #     except ValueError:
            #         # 如果 (x, y) 不在列表中，index 方法会抛出 ValueError
            #         self.logger.error(f'({x},{y}) is not a valid deployment point.')

            else:
                self.logger.info(f'{unit_name} is deployed to ({x},{y}).' + '\n' + f'angle from {angle[0]} to {angle[1]}')
        
            # if unit_name.startswith('flak'):
            #     self.red_team.append(
            #         StoneFlak(name=unit_name,
            #                   posx=x,
            #                   posy=y,
            #                   reward=-1,
            #                   health=self.config['flak_health'],
            #                   attack=self.config['flak_attack'],
            #                   attack_range=self.config['flak_attack_range'],
            #                   accuracy=self.config['flak_accuracy'],
            #                   ammunition=self.config['flak_ammunition']))
            if unit_name.startswith('flak'):
                # 三个位置写死的防空炮台
                if temp_Flak_num == 0:
                    temp_Flak_num = temp_Flak_num + 1
                    self.red_team.append(
                        StoneFlak(name=unit_name,
                                    posx=76,
                                    posy=75,
                                    reward=-1,
                                    health=self.config['flak_health'],
                                    attack=self.config['flak_attack'],
                                    attack_range=self.config['flak_attack_range'],
                                    accuracy=self.config['flak_accuracy'],
                                    ammunition=self.config['flak_ammunition']))
                elif temp_Flak_num == 1:
                    temp_Flak_num = temp_Flak_num + 1
                    self.red_team.append(
                        StoneFlak(name=unit_name,
                                    posx=88,
                                    posy=75,
                                    reward=-1,
                                    health=self.config['flak_health'],
                                    attack=self.config['flak_attack'],
                                    attack_range=self.config['flak_attack_range'],
                                    accuracy=self.config['flak_accuracy'],
                                    ammunition=self.config['flak_ammunition']))
                elif temp_Flak_num == 2:
                    temp_Flak_num = temp_Flak_num + 1
                    self.red_team.append(
                        StoneFlak(name=unit_name,
                                    posx=98,
                                    posy=70,
                                    reward=-1,
                                    health=self.config['flak_health'],
                                    attack=self.config['flak_attack'],
                                    attack_range=self.config['flak_attack_range'],
                                    accuracy=self.config['flak_accuracy'],
                                    ammunition=self.config['flak_ammunition']))
        
            # elif unit_name.startswith('near_turrer'):
            #     # 设置 near_turret 的 angle
            #     if angle in {0, 1, 2, 3}:
            #         cur_angle = 180
            #     elif angle in {4, 5, 6, 7}:
            #         cur_angle = 270
            #     else:
            #         cur_angle = 0  # 默认值或处理未知方向
            #     self.red_team.append(
            #         StoneNearTurret(
            #             name=unit_name,
            #             posx=x,
            #             posy=y,
            #             direction=[directions],
            #             reward=-1,
            #             health=self.config['near_turret_health'],
            #             attack=self.config['near_turret_attack'],
            #             # angle=self.config['near_turret_angle'],
            #             angle=cur_angle,
            #             attack_range=self.config['near_turret_attack_range'],
            #             accuracy=self.config['near_turret_accuracy'],
            #             ammunition=self.config['near_turret_ammunition']))
            elif unit_name.startswith('near_turret'):
                self.red_team.append(
                    StoneNearTurret(
                        name=unit_name,
                        posx=x,
                        posy=y,
                        
                        # direction=[directions],
                        reward=-1,
                        health=self.config['near_turret_health'],
                        attack=self.config['near_turret_attack'],
                        # angle=self.config['near_turret_angle'],
                        angle_start=angle[0],
                        angle_end=angle[1],
                        attack_range=self.config['near_turret_attack_range'],
                        accuracy=self.config['near_turret_accuracy'],
                        ammunition=self.config['near_turret_ammunition']))
                

            elif unit_name.startswith('far_turret'):
                self.red_team.append(
                    StoneFarTurret(
                        name=unit_name,
                        posx=x,
                        posy=y,
                        
                        reward=-1,
                        health=self.config['far_turret_health'],
                        attack=self.config['far_turret_attack'],
                        angle_start=angle[0],
                        angle_end=angle[1],
                        attack_range=self.config['far_turret_attack_range'],
                        accuracy=self.config['far_turret_accuracy'],
                        ammunition=self.config['far_turret_ammunition']))
        self.update_map()

    def update_map(self):
        if self.is_render:
            self.map.update(
                self.red_team + self.blue_team + self.blue_team_preparation,
                self.step)

    def simulate_battle(self):
        result = -1
        self.update_map()
        while True:
            self.step += 1
            formatter = logging.Formatter(
                f'%(asctime)s | %(levelname)s | Step {self.step} | %(message)s'
            )
            self.console_handler.setFormatter(formatter)
            self.file_handler.setFormatter(formatter)

            # Red team attack
            # red_attack = {}
            # for red_unit in self.red_team:
            #     if red_unit.get_states() != 0:
            #         target, damage = red_unit.attack_check(self.blue_team)
            #         if target is not None:
            #             self.logger.info(
            #                 f'{red_unit.get_name()} deal {damage} damage to {target}.'
            #             )
            #             if target in red_attack:
            #                 red_attack[target] += damage
            #             else:
            #                 red_attack[target] = damage

            red_attack = []
            health_record = {}
            for blue_unit in self.blue_team:
                health_record[blue_unit.get_name()]=blue_unit.get_health()
            
                
            for red_unit in self.red_team:
                if red_unit.get_states() != 0:
                    # print(f'red_unit:{red_unit.name}')
                    # visible_units = self.map.filter_visible_units(red_unit, self.blue_team)
                    targets, total_damage = red_unit.attack_check(self.blue_team,health_record)
                    # if targets and total_damage:  # 检查是否有目标和伤害
                    # print(f'targets:{targets}')
                    # print(f'total_damage:{total_damage}')
                    # 假设 targets 和 total_damage 可能是列表，也可能是单个值
                    if not isinstance(targets, list):
                        targets = [targets]
                    if not isinstance(total_damage, list):
                        total_damage = [total_damage]
                    # print(f'targets:{targets}')
                    # print(f'total_damage:{total_damage}')
                    for target, damage in zip(targets, total_damage):
                        if target is not None:
                            self.logger.info(
                                f'{red_unit.get_name()} deal {damage} damage to {target}.'
                            )
                            red_attack.append((target, damage))
                            
                


            # print(f'self.deployment_points:{self.deployment_points}')

            # # Red team's attack applies to Blue team
            # for blue_unit in self.blue_team:
            #     unit_name = blue_unit.get_name()
            #     if unit_name in red_attack:
            #         blue_unit.set_damage(red_attack[unit_name])

            for blue_unit in self.blue_team:
                unit_name = blue_unit.get_name()

                # 检查是否存在指定的士兵名字
                exists = any(soldier == unit_name for soldier, value in red_attack)
                # print(f"Does {unit_name} exist in the list? {exists}")
                
                # 如果存在则求和，否则返回0或其他默认值
                if exists:
                    total_damage = sum(value for soldier, value in red_attack if soldier == unit_name)
                    blue_unit.set_damage(total_damage)
            

            # Blue team depoly
            for _ in range(self.config['unit_num_per_step']):
                if len(self.blue_team_preparation) > 0:
                    blue_unit = self.blue_team_preparation.pop()
                    self.blue_team.append(blue_unit)

            self.update_map()

            # Blue team move
            for blue_unit in self.blue_team:
                if blue_unit.get_states() == 1:
                    blue_unit.move()

            self.update_map()

            # Blue team attack
            blue_attack = {}
            for blue_unit in self.blue_team:
                if blue_unit.get_states() not in [0, 3]:
                    # visible_units = self.map.filter_visible_units(blue_unit, self.red_team)
                    target, damage = blue_unit.attack_check(self.red_team)
                    if target is not None:
                        self.logger.info(
                            f'{blue_unit.get_name()} deal {damage} damage to {target}.'
                        )
                        if target in blue_attack:
                            blue_attack[target] += damage
                        else:
                            blue_attack[target] = damage

            # Blue team's attack applies to Red team
            for red_unit in self.red_team:
                unit_name = red_unit.get_name()
                if unit_name in blue_attack:
                    red_unit.set_damage(blue_attack[unit_name])

            self.update_map()

            # summary
            red_dead = 0
            red_alive_name, red_dead_name = [], []
            blue_dead, blue_evacuated = 0, 0
            blue_alive_name, blue_dead_name, blue_evacuated_name = [], [], []
            blue_team_sum_distance = 0
            for blue_unit in self.blue_team:
                blue_unit_states = blue_unit.get_states()
                blue_unit_name = blue_unit.get_name()
                if blue_unit_states == 0:
                    blue_dead += 1
                    blue_dead_name.append(blue_unit_name)
                elif blue_unit_states == 3:
                    blue_evacuated += 1
                    blue_evacuated_name.append(blue_unit_name)
                else:
                    blue_alive_name.append(blue_unit_name)
                if blue_unit_states == 1:
                    blue_team_sum_distance += blue_unit.get_remaining_distance(
                    )

            for red_unit in self.red_team:
                if red_unit.get_states() == 0:
                    red_dead += 1
                    red_dead_name.append(red_unit.get_name())
                else:
                    red_alive_name.append(red_unit.get_name())

            self.logger.info('Red team alive: ' +
                             ','.join(sorted(red_alive_name)))
            self.logger.info('Red team dead: ' +
                             ','.join(sorted(red_dead_name)))
            self.logger.info('Blue team alive: ' +
                             ','.join(sorted(blue_alive_name)))
            self.logger.info('Blue team evacuated: ' +
                             ','.join(sorted(blue_evacuated_name)))
            self.logger.info('Blue team dead: ' +
                             ','.join(sorted(blue_dead_name)))

            # Endding
            if self.step > 300:
                self.logger.fatal('Timeout')
                result = 0
                break
            if self.blue_team_unit_num - blue_dead -10< self.config[
                    'minimum_unit_evacuated']:
                self.logger.fatal(
                    'The blue team unable to evacuate enough units.')
                result = 1
                break
            if blue_evacuated >= self.config['minimum_unit_evacuated']:
                self.logger.fatal('The blue team has been evacuated.')
                result = 2
                break
            if red_dead == self.red_team_unit_num:
                self.logger.fatal('The red team is eliminated.')
                result = 3
                break

        return result, blue_dead, blue_evacuated, blue_team_sum_distance, red_dead



if __name__ == "__main__":
    deployment_points_positions = [(74, 60), (74, 62), (74, 64), (74, 73), (74, 75), (74, 77), (76, 60), (76, 64), (76, 73), (76, 77), (78, 60), (78, 62), (78, 64), (78, 73), (78, 75), (78, 77), (83, 68), (83, 70), (83, 72), (85, 68), (85, 72), (87, 68), (87, 70), (87, 72), (96, 60), (96, 62), (96, 64), (96, 73), (96, 75), (96, 77), (98, 60), (98, 64), (98, 73), (98, 77), (100, 60), (100, 62), (100, 64), (100, 73), (100, 75), (100, 77)]
    map = Map(save_path='result.log', is_render=False)
    map.setup_map() 
    visibility_map = map.generate_visibility_map(deployment_points_positions)
    print('visibility_map' + f"{visibility_map}")
    map.save_visibility_map_to_json(visibility_map)
    