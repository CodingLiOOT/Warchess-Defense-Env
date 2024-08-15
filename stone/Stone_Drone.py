import math
import random
import pygame as pg

from .Basestone import BaseStone


class StoneDrone(BaseStone):
    '''
    子类
    attack_arrage是一个长度为2的list。分别代表最近打击距离和最远打击距离
    accuracy为攻击命中率
    '''

    def __init__(self,
                 name,
                 posx,
                 posy,
                 route,
                 reward,
                 mobility=2,
                 health=5,
                 attack=1,
                 attack_range=[0, 5],
                 accuracy=0.9,
                 ammunition=5):
        super().__init__(health=health,
                         mobility=mobility,
                         stone_type='drone',
                         name=name,
                         reward=reward,
                         posx=posx,
                         posy=posy,
                         route=route,
                         attack=attack,
                         accuracy=accuracy,
                         attack_range=attack_range,
                         ammunition=ammunition)

        self.image = pg.image.load('image/icons8-drone-64.png')
        self.color = [(153, 204, 255, 128), (102, 178, 255, 128),
                      (51, 153, 255, 128), (0, 128, 255, 128),
                      (0, 102, 204, 128), (0, 76, 153, 128), (0, 51, 102, 128),
                      (0, 25, 51, 128), (0, 0, 255, 128), (0, 0, 153, 128)]

    def is_enemy_in_range(self, ex, ey):
        distance = math.sqrt(((self.posx - ex)**2 + (self.posy - ey)**2))
        return self.attack_range[0] <= distance <= self.attack_range[1]

    def attack_check(self, enemys):
        '''
        找到enemys列表中在攻击范围内且最近的敌人，并对其进行打击。

        参数:
        enemys (list of objects): 红方的炮塔列表，每个单位需具有posx, posy, name, 和 health 属性。

        输出:
        被攻击单位的name，对其造成的伤害。
        '''
        min_distance = float('inf')  # 初始化最小距离为无穷大
        target = None  # 初始化目标敌人
        damage = 0  # 造成的伤害

        # 检查弹药量，不为0时才进行攻击
        if self.ammunition != 0:
            # 遍历所有炮塔，寻找最近的目标
            for enemy in enemys:
                if enemy.get_states() in [0]:  # 跳过已死亡的敌人
                    continue
                # 攻击距离自身最近的敌人
                x, y = enemy.get_pos()
                if self.is_enemy_in_range(x, y):
                    distance = math.sqrt(
                        ((self.posx - x)**2 + (self.posy - y)**2))
                    if distance < min_distance:
                        min_distance = distance
                        target = enemy.get_name()

            # 如果找到目标则消耗弹药，并进行伤害判断
            if target is not None:
                self.ammunition = self.ammunition - 1
                hit = random.uniform(0, 1) < self.accuracy
                damage = self.attack if hit else 0

        return target, damage
