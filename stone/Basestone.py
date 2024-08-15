import logging

logger = logging.getLogger()


class BaseStone:
    """
    棋子类的基类
    Args:
        stone_type: 士兵是何种类型
        name: 棋子的名字
        mobility: 棋子的行动力
        states: 棋子的状态:0:死亡 1:可移动 2:不可移动 3:已撤离
        reward: 表示棋子的价值
        posx:棋子的X轴坐标 [0,199]
        posy:棋子的Y轴坐标 [0,59]
        direction: 单位朝向,主要用于攻击,例如[0]表示360°,其他数值表示其8邻域方向
        route: 移动时的direction list
        ammunition:弹药量,-1为无限弹药
    """

    def __init__(
        self,
        name='',
        stone_type='',
        posx=0,
        posy=0,
        route=[],
        reward=-1,
        mobility=1,
        health=10,
        attack=1,
        attack_range=[0, 5],
        ammunition=-1,
        direction=[-1],
        accuracy=0.9,
    ):
        self.name = name
        self.stone_type = stone_type

        self.image = None
        self.color = None

        self.health = health
        self.attack = attack
        self.attack_range = attack_range
        self.accuracy = accuracy
        self.ammunition = ammunition

        # 棋子的状态:0:死亡 1:可移动 2:不可移动 3:已撤离
        self.states = 1 if mobility > 0 else 2
        if self.states == 1:
            self.mobility = mobility
            self.route = route
            self.direction = direction

        if self.states != 1:
            self.mobility = 0
            self.route = []
            self.direction = [-1]

        self.reward = reward
        self.posx = posx
        self.posy = posy

    def set_states(self, states):
        self.states = states

        if self.states != 1:
            self.mobility = 0
            self.route = []
            self.direction = [-1]

        if self.states == 3:
            logger.info(f'{self.name} has been evacuated.')

    def set_damage(self, damage):
        self.health = self.health - damage

        if self.health <= 0:
            self.set_states(0)
            logger.info(f'{self.name} is destroyed.')

    def set_direction(self, dir):
        self.direction = dir

    def get_health(self):
        return self.health

    def get_direction(self):
        return self.direction

    def get_stone_type(self):
        return self.stone_type

    def get_name(self):
        return self.name

    def get_color(self, idx=None):
        if idx is None:
            return self.color
        else:
            idx = -1 if idx > len(self.color) else idx
            return self.color[idx]

    def get_image(self):
        return self.image

    def get_states(self):
        return self.states

    def get_mobility(self):
        return self.mobility

    def get_reward(self):
        return self.reward

    def get_pos(self):
        return self.posx, self.posy

    def get_posx(self):
        return self.posx

    def get_posy(self):
        return self.posy

    def get_remaining_distance(self):
        if self.states != 1:
            return float('inf')
        else:
            return len(self.route)

    def set_pos(self, posx, posy):
        self.posx = posx
        self.posy = posy

    '''
    8 1 2
    7 0 3
    6 5 4
    '''

    def _move(self, direct):
        if self.mobility > 0:
            if direct == 1:
                self.posy = self.posy - 1
            elif direct == 2:
                self.posx = self.posx + 1
                self.posy = self.posy - 1
            elif direct == 3:
                self.posx = self.posx + 1
            elif direct == 4:
                self.posx = self.posx + 1
                self.posy = self.posy + 1
            elif direct == 5:
                self.posy = self.posy + 1
            elif direct == 6:
                self.posx = self.posx - 1
                self.posy = self.posy + 1
            elif direct == 7:
                self.posx = self.posx - 1
            elif direct == 8:
                self.posx = self.posx - 1
                self.posy = self.posy - 1

    def move(self):
        for _ in range(self.mobility):
            if len(self.route) > 0:
                self._move(self.route.pop(0))
            else:
                self.set_states(3)
        logger.info(f'{self.name} move to ({self.posx}, {self.posy}).')
