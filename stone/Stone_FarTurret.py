import math
import random
import pygame as pg

from .Basestone import BaseStone


class StoneFarTurret(BaseStone):
    '''
    防御塔类
    attack_range是一个长度为2的list，分别代表最近打击距离和最远打击距离
    accuracy为攻击命中率
    directions代表防御塔的朝向（可以是一个包含多个朝向的list）
    angle代表攻击角度
    '''

    DIRECTION_ANGLES = {
        0: 270,  # 上
        1: 315,  # 右上
        2: 0,  # 右
        3: 45,  # 右下
        4: 90,  # 下
        5: 135,  # 左下
        6: 180,  # 左
        7: 225,  # 左上
    }

    def __init__(
        self,
        name,
        posx,
        posy,
        reward,
        health=2000,
        attack=1,
        angle_start=20,
        angle_end=180,
        attack_range=[3, 10],
        direction=[0],
        accuracy=0.9,
        ammunition=100,
    ):
        super().__init__(
            health=health,
            mobility=0,
            stone_type='far_turret',
            name=name,
            reward=reward,
            posx=posx,
            posy=posy,
            route=[],
            attack=attack,
            accuracy=accuracy,
            attack_range=attack_range,
            ammunition=ammunition,
        )

        self.direction = direction  # 朝向列表
        self.angle_start = angle_start  # 攻击起始边界角度
        self.angle_end = angle_end  # 攻击最终边界角度
        # 根据角度差值计算每回合攻击次数
        angle_diff = abs(self.angle_end - self.angle_start)
        self.attacks_per_turn = max(1, 5 - (angle_diff // 72))

        self.image = pg.image.load('image/icons8-mortar-96.png')
        self.color = [(204, 0, 0, 128)]

    # def is_enemy_in_range(self, ex, ey):
    #     dx = ex - self.posx
    #     dy = ey - self.posy
    #     enemy_distance = math.sqrt(dx * dx + dy * dy)

    #     # 如果敌人距离大于防御塔的最远攻击距离或小于最近攻击距离，则敌人在攻击范围外
    #     if not (self.attack_range[0] <= enemy_distance <=
    #             self.attack_range[1]):
    #         return False

    #     # 计算敌人相对于防御塔的角度
    #     enemy_angle = math.degrees(math.atan2(dy, dx))
    #     if enemy_angle < 0:
    #         enemy_angle += 360  # 确保角度在0到360度之间

    #     for d in self.direction:
    #         # 将防御塔的朝向转换为角度
    #         tower_direction_angle = self.DIRECTION_ANGLES[d]
    #         relative_angle = enemy_angle - tower_direction_angle

    #         # 将角度标准化到[-180, 180]
    #         if relative_angle < -180:
    #             relative_angle += 360
    #         if relative_angle > 180:
    #             relative_angle -= 360

    #         # 计算攻击角度的一半
    #         half_angle = self.angle / 2

    #         # 检查敌人是否在攻击角度范围内
    #         if -half_angle <= relative_angle <= half_angle:
    #             return True

    #     return False

    # def is_enemy_in_range(self, ex, ey):
    #     dx = ex - self.posx
    #     dy = ey - self.posy
    #     enemy_distance = math.sqrt(dx * dx + dy * dy)

    #     # 如果敌人距离大于防御塔的最远攻击距离或小于最近攻击距离，则敌人在攻击范围外
    #     if not (self.attack_range[0] <= enemy_distance <=
    #             self.attack_range[1]):
    #         return False

    #     # 计算敌人相对于防御塔的角度
    #     enemy_angle = math.degrees(math.atan2(dy, dx))
    #     if enemy_angle < 0:
    #         enemy_angle += 360  # 确保角度在0到360度之间

    #     for d in self.direction:
    #         # 将防御塔的朝向转换为角度
    #         tower_direction_angle = self.DIRECTION_ANGLES[d]
    #         relative_angle = enemy_angle - tower_direction_angle

    #         # 将角度标准化到[-180, 180]
    #         if relative_angle < -180:
    #             relative_angle += 360
    #         if relative_angle > 180:
    #             relative_angle -= 360

    #         # 计算攻击角度的一半
    #         half_angle = self.angle / 2

    #         # 检查敌人是否在攻击角度范围内
    #         if -half_angle <= relative_angle <= half_angle:
    #             return True

    #     return False

    # def is_enemy_in_range(self, ex, ey):
    #     distance = math.sqrt(((self.posx - ex)**2 + (self.posy - ey)**2))
    #     return self.attack_range[0] <= distance <= self.attack_range[1]

    # def is_enemy_in_range(self, ex, ey):
    #     dx = ex - self.posx
    #     dy = ey - self.posy
    #     enemy_distance = math.sqrt(dx * dx + dy * dy)

    #     # 如果敌人距离大于防御塔的最远攻击距离或小于最近攻击距离，则敌人在攻击范围外
    #     if not (self.attack_range[0] <= enemy_distance <=
    #             self.attack_range[1]):
    #         return False

    #     # 计算敌人相对于防御塔的角度
    #     enemy_angle = math.degrees(math.atan2(dy, dx))
    #     if enemy_angle < 0:
    #         enemy_angle += 360  # 确保角度在0到360度之间

    #     for d in self.direction:
    #         # 将防御塔的朝向转换为角度
    #         tower_direction_angle = self.DIRECTION_ANGLES[d]
    #         relative_angle = enemy_angle - tower_direction_angle

    #         # 将角度标准化到[-180, 180]
    #         if relative_angle < -180:
    #             relative_angle += 360
    #         if relative_angle > 180:
    #             relative_angle -= 360

    #         # 计算攻击角度的一半
    #         half_angle = self.angle / 2

    #         # 检查敌人是否在攻击角度范围内
    #         if -half_angle <= relative_angle <= half_angle:
    #             return True

    #     return False

    # def is_enemy_in_range(self, ex, ey):
    #     dx = ex - self.posx
    #     dy = ey - self.posy
    #     enemy_distance = math.sqrt(dx * dx + dy * dy)

    #     # 如果敌人距离大于防御塔的最远攻击距离或小于最近攻击距离，则敌人在攻击范围外
    #     if not (self.attack_range[0] <= enemy_distance <= self.attack_range[1]):
    #         return False

    #     # 计算敌人相对于防御塔的角度
    #     enemy_angle = math.degrees(math.atan2(dy, dx))
    #     if enemy_angle < 0:
    #         enemy_angle += 360  # 确保角度在0到360度之间

    #     for d in self.direction:
    #         # 将防御塔的朝向转换为角度
    #         tower_direction_angle = self.DIRECTION_ANGLES[d]
    #         relative_angle = enemy_angle - tower_direction_angle

    #         # 将角度标准化到[-180, 180]
    #         if relative_angle < -180:
    #             relative_angle += 360
    #         if relative_angle > 180:
    #             relative_angle -= 360

    #         # 检查敌人是否在攻击角度范围内
    #         if self.angle_start <= relative_angle <= self.angle_end:
    #             return True

    #     return False

    def is_enemy_in_range(self, ex, ey):
        dx = ex - self.posx
        dy = ey - self.posy
        enemy_distance = math.sqrt(dx * dx + dy * dy)

        # 如果敌人距离大于防御塔的最远攻击距离或小于最近攻击距离，则敌人在攻击范围外
        if not (self.attack_range[0] <= enemy_distance <= self.attack_range[1]):
            return False

        # 计算敌人相对于防御塔的角度
        enemy_angle = math.degrees(
            math.atan2(dx, -dy)
        )  # 调整计算方式，以Y轴正方向为0度，顺时针为正
        if enemy_angle < 0:
            enemy_angle += 360  # 确保角度在0到360度之间

        # 标准化角度范围
        angle_start = self.angle_start % 360
        angle_end = self.angle_end % 360

        # 检查敌人是否在攻击角度范围内
        if angle_start < angle_end:
            return angle_start <= enemy_angle <= angle_end
        else:
            return enemy_angle >= angle_start or enemy_angle <= angle_end

    # def attack_check(self, enemys):
    #     '''
    #     找到enemys列表中在攻击范围内且最近的敌人，并对其进行打击。

    #     参数:
    #     enemys (list of objects): 敌人的列表，每个单位需具有posx, posy, name, 和 health 属性。

    #     输出:
    #     被攻击单位的name，对其造成的伤害。
    #     '''
    #     min_distance = float('inf')  # 初始化最小距离为无穷大
    #     target = None  # 初始化目标敌人
    #     damage = 0  # 造成的伤害

    #     # 检查弹药量，不为0时才进行攻击
    #     if self.ammunition != 0:
    #         # 遍历所有炮塔，寻找最近的目标
    #         for enemy in enemys:
    #             if enemy.get_states() in [0, 3]:  # 跳过已死亡与已撤离的敌人
    #                 continue
    #             if enemy.get_stone_type() in ['drone']:  # 跳过无人机
    #                 continue
    #             x, y = enemy.get_pos()
    #             # 攻击距离撤离点最近的敌人
    #             if self.is_enemy_in_range(x, y):
    #                 distance = enemy.get_remaining_distance()
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     target = enemy.get_name()

    #         # 如果找到目标则消耗弹药，并进行伤害判断
    #         if target is not None:
    #             self.ammunition = self.ammunition - 1
    #             hit = random.uniform(0, 1) < self.accuracy
    #             damage = self.attack if hit else 0

    #     return target, damage

    def attack_check(self, enemys, health_record):
        '''
        找到enemys列表中在攻击范围内且最近的敌人，并对其进行多次打击。

        参数:
        enemys (list of objects): 敌人的列表，每个单位需具有posx, posy, name, 和 health 属性。

        输出:
        被攻击单位的name列表，对其造成的伤害列表。
        '''
        total_damage = []
        targets = []

        # 检查弹药量，确保有足够的弹药进行多次攻击
        if self.ammunition != 0:
            for _ in range(self.attacks_per_turn):
                min_distance = float('inf')  # 初始化最小距离为无穷大
                target = None  # 初始化目标敌人

                for enemy in enemys:
                    if enemy.get_states() in [0, 3]:  # 跳过已死亡与已撤离的敌人
                        continue
                    if enemy.get_stone_type() in ['drone']:  # 跳过无人机
                        continue
                    if health_record[enemy.get_name()] <= 0:
                        continue
                    x, y = enemy.get_pos()
                    # 攻击距离撤离点最近的敌人
                    if self.is_enemy_in_range(x, y):
                        distance = enemy.get_remaining_distance()
                        if distance < min_distance:
                            min_distance = distance
                            target = enemy

                # 如果找到目标则消耗弹药，并进行伤害判断
                if target is not None and self.ammunition > 0:
                    self.ammunition -= 1
                    hit = random.uniform(0, 1) < self.accuracy
                    damage = self.attack if hit else 0
                    total_damage.append(damage)
                    targets.append(target.get_name())
                    health_record[target.get_name()] -= damage

        # print(f'targets:{targets}')
        # print(f'total_damage:{total_damage}')
        return targets, total_damage
