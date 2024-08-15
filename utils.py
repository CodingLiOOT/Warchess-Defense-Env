import random
import numpy as np

# #################################################
# Map
# #################################################

x_len = 8
y_len = 8
image_x = 15
image_y = 15

# load map
map = []
with open('resource/map.csv', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        map.append([int(x) for x in line])

map_width = len(map[0])
map_height = len(map)

# #################################################
# Route
# #################################################


def get_routing():
    with open('resource/ground_routing.csv', 'r') as f:
        ground_routing = []
        for line in f.readlines():
            line = line.strip().split(',')
            line = [int(x) for x in line]
            ground_routing.append((line[:2], line[2:]))
    random.shuffle(ground_routing)

    with open('resource/aerial_routing.csv', 'r') as f:
        aerial_routing = []
        for line in f.readlines():
            line = line.strip().split(',')
            line = [int(x) for x in line]
            aerial_routing.append((line[:2], line[2:]))
    random.shuffle(aerial_routing)
    return ground_routing, aerial_routing


def unit_move(x, y, direct):
    if direct == 1:
        y = y - 1
    elif direct == 2:
        x = x + 1
        y = y - 1
    elif direct == 3:
        x = x + 1
    elif direct == 4:
        x = x + 1
        y = y + 1
    elif direct == 5:
        y = y + 1
    elif direct == 6:
        x = x - 1
        y = y + 1
    elif direct == 7:
        x = x - 1
    elif direct == 8:
        x = x - 1
        y = y - 1
    return (x, y)


def get_move_sequence(x, y, x2, y2):
    directions = []
    path = []

    while (x, y) != (x2, y2):
        if random.random() < 0.5:
            if x < x2 and y < y2:
                direction = 4  # Move diagonally down-right
            elif x < x2 and y > y2:
                direction = 2  # Move diagonally up-right
            elif x > x2 and y < y2:
                direction = 6  # Move diagonally down-left
            elif x > x2 and y > y2:
                direction = 8  # Move diagonally up-left
            elif x < x2:
                direction = 3  # Move right
            elif y > y2:
                direction = 1  # Move up
            elif x > x2:
                direction = 7  # Move left
            elif y < y2:
                direction = 5  # Move down
        else:
            if x > x2 and y > y2:
                direction = 8  # Move diagonally up-left
            elif x < x2 and y < y2:
                direction = 4  # Move diagonally down-right
            elif x < x2 and y > y2:
                direction = 2  # Move diagonally up-right
            elif x > x2 and y < y2:
                direction = 6  # Move diagonally down-left
            elif x > x2:
                direction = 7  # Move left
            elif y < y2:
                direction = 5  # Move down
            elif x < x2:
                direction = 3  # Move right
            elif y > y2:
                direction = 1  # Move up
        # if random.random() < 0.5:
        #     if x < x2:
        #         direction = 3  # Move right
        #     elif x > x2:
        #         direction = 7  # Move left
        #     elif y > y2:
        #         direction = 1  # Move up
        #     elif y < y2:
        #         direction = 5  # Move down
        # else:
        #     if y > y2:
        #         direction = 1  # Move up
        #     elif y < y2:
        #         direction = 5  # Move down
        #     elif x < x2:
        #         direction = 3  # Move right
        #     elif x > x2:
        #         direction = 7  # Move left
        directions.append(direction)
        path.append((x, y))
        x, y = unit_move(x, y, direction)
    assert x == x2 and y == y2, "src:{}, dst:{}\npath:{}".format((x, y), (x2, y2), path)
    return directions, path


# #################################################
# function
# #################################################


def get_hex_map_pos(x, y):
    if y % 2 == 0:
        base_x = x_len * 2 * x
        base_y = y_len * 3 * (y // 2)
    else:
        base_x = x_len * 2 * x + x_len
        base_y = y_len * 3 * (y // 2) + y_len // 2 + y_len
    return base_x, base_y


def idx2xy(num):
    x = num % map_width
    y = num // map_width
    return x, y


def xy2idx(x, y):
    return x + y * map_width


def comput_xy(x, y):
    w = int(x / 2)
    dline = len(map[0]) + len(map[1])
    if x % 2 == 0:
        return w * dline + y
    else:
        return w * dline + y + map_width


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def mix_colors(colors):
    """
    混合多个颜色，根据各自的数量进行加权平均。
    colors: [(color, count), ...]
    color: (R, G, B)
    count: 该颜色单位的数量
    """
    total_count = sum(count for _, count in colors)
    if total_count == 0:
        return (0, 0, 0)

    mixed_color = [0, 0, 0]

    for color, count in colors:
        for i in range(3):  # R, G, B
            mixed_color[i] += color[i] * count

    mixed_color = [int(c / total_count) for c in mixed_color]

    return tuple(mixed_color)
