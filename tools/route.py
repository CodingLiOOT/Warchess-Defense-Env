# # # %%
# # from email import utils
# # import sys
# # import os
# # import json
# # root_dir = os.path.dirname(os.path.dirname(__file__))
# # root_dir = '.' if len(root_dir) == 0 else root_dir
# # os.chdir(root_dir)
# # if root_dir not in sys.path:
# #     sys.path.insert(0, root_dir)
# # # %%
# # import random
# # import utils as t


# # # %%
# # #       8 1 2
# # #       7 0 3
# # #       6 5 4
# # def _move(x, y, direct):
# #     if direct == 1:
# #         y = y - 1
# #     elif direct == 2:
# #         x = x + 1
# #         y = y - 1
# #     elif direct == 3:
# #         x = x + 1
# #     elif direct == 4:
# #         x = x + 1
# #         y = y + 1
# #     elif direct == 5:
# #         y = y + 1
# #     elif direct == 6:
# #         x = x - 1
# #         y = y + 1
# #     elif direct == 7:
# #         x = x - 1
# #     elif direct == 8:
# #         x = x - 1
# #         y = y - 1
# #     return (x, y)


# # def get_move_sequence(x, y, x2, y2):
# #     directions = []
# #     path = []

# #     while (x, y) != (x2, y2):
# #         if random.random() < 0.5:
# #             if x < x2 and y < y2:
# #                 direction = 4  # Move diagonally down-right
# #             elif x < x2 and y > y2:
# #                 direction = 2  # Move diagonally up-right
# #             elif x > x2 and y < y2:
# #                 direction = 6  # Move diagonally down-left
# #             elif x > x2 and y > y2:
# #                 direction = 8  # Move diagonally up-left
# #             elif x < x2:
# #                 direction = 3  # Move right
# #             elif y > y2:
# #                 direction = 1  # Move up
# #             elif x > x2:
# #                 direction = 7  # Move left
# #             elif y < y2:
# #                 direction = 5  # Move down
# #         else:
# #             if x > x2 and y > y2:
# #                 direction = 8  # Move diagonally up-left
# #             elif x < x2 and y < y2:
# #                 direction = 4  # Move diagonally down-right
# #             elif x < x2 and y > y2:
# #                 direction = 2  # Move diagonally up-right
# #             elif x > x2 and y < y2:
# #                 direction = 6  # Move diagonally down-left
# #             elif x > x2:
# #                 direction = 7  # Move left
# #             elif y < y2:
# #                 direction = 5  # Move down
# #             elif x < x2:
# #                 direction = 3  # Move right
# #             elif y > y2:
# #                 direction = 1  # Move up
# #         # if random.random() < 0.5:
# #         #     if x < x2:
# #         #         direction = 3  # Move right
# #         #     elif x > x2:
# #         #         direction = 7  # Move left
# #         #     elif y > y2:
# #         #         direction = 1  # Move up
# #         #     elif y < y2:
# #         #         direction = 5  # Move down
# #         # else:
# #         #     if y > y2:
# #         #         direction = 1  # Move up
# #         #     elif y < y2:
# #         #         direction = 5  # Move down
# #         #     elif x < x2:
# #         #         direction = 3  # Move right
# #         #     elif x > x2:
# #         #         direction = 7  # Move left
# #         directions.append(direction)
# #         path.append((x, y))
# #         x, y = _move(x, y, direction)
# #     assert (x == x2 and y == y2), "src:{}, dst:{}\npath:{}".format(
# #         (x, y), (x2, y2), path)
# #     return directions, path


# # # %%


# # def is_away_from_end(p1, p2, end):
# #     """
# #     检查从当前点移动到下一个点是否远离终点。
    
# #     :param p1: 当前点 (x, y)
# #     :param p2: 下一个点 (x, y)
# #     :param end: 终点 (x, y)
# #     :return: 如果远离终点返回 True，否则返回 False
# #     """
# #     dist_p1_end = t.euclidean_distance(p1, end)
# #     dist_p2_end = t.euclidean_distance(p2, end)
# #     return dist_p2_end > dist_p1_end


# # def is_between(p1, p2, p):
# #     return ((p1[0] <= p[0] <= p2[0] or p2[0] <= p[0] <= p1[0])
# #             and (p1[1] <= p[1] <= p2[1] or p2[1] <= p[1] <= p1[1]))


# # def is_reachable(p1, p2, end=None, is_drone=False):

# #     # # 限制无人机移动，减少搜索空间
# #     # if is_drone:
# #     #     return random.random() > 0.5

# #     # 检查通过路径中是否存在障碍
# #     flag = True
# #     _, path = get_move_sequence(p1[0], p1[1], p2[0], p2[1])
# #     for p in path:
# #         if p == end:  # 如果路径途径终点，必须确保是最终抵达终点
# #             return False
# #         flag = flag and ((t.map[p[1]][p[0]] not in [1, 2]) or is_drone)
# #     return flag


# # # %%
# # def get_one_path(start, end, waypoints, is_drone=False):

# #     def dfs(current, end, waypoints, path):
# #         if current == end and len(path) > 1:
# #             return path

# #         waypoints = [
# #             p for p in waypoints if not is_away_from_end(current, p, end)
# #         ]
# #         random.shuffle(waypoints)
# #         for i, waypoint in enumerate(waypoints):
# #             if waypoint not in path and is_reachable(current, waypoint, end,
# #                                                      is_drone):
# #                 path.append(waypoint)
# #                 remaining_waypoints = waypoints[:i] + waypoints[i + 1:]
# #                 result = dfs(waypoint, end, remaining_waypoints, path)
# #                 if result:
# #                     return result
# #                 path.pop()

# #         return None

# #     path = [start]
# #     return dfs(start, end, waypoints, path)


# # def get_all_route_sequence(waypoints,
# #                            entrances_points,
# #                            exit_points,
# #                            num=5,
# #                            is_drone=False):  # -> tuple[list, list]:
# #     route_list = []
# #     for entrances in entrances_points:
# #         _route_list = []
# #         for exit in exit_points:
# #             while len(_route_list) < num:
# #                 route = get_one_path(entrances, exit, waypoints, is_drone)
# #                 _str_route = str(route)
# #                 if _str_route not in _route_list:
# #                     _route_list.append(_str_route)
# #                     route_list.append(route)

# #     sequence = []
# #     for route in route_list:
# #         ds, rs = [], []
# #         c = route.pop(0)
# #         while len(route) > 0:
# #             n = route.pop(0)
# #             _ds, _rs = get_move_sequence(c[0], c[1], n[0], n[1])
# #             rs = rs + _rs
# #             ds = ds + _ds
# #             c = n
# #         sequence.append([ds, rs])
# #     return sequence

# # def convert_structure(data):
# #     from collections import defaultdict
# #     path_dict = defaultdict(list)
# #     for item in data:
# #         main_data = item[0]
# #         start = item[1][0]
# #         path_dict[start].append(main_data)
# #     result = {"routes":[]}
# #     for start,paths in path_dict.items():
# #         route={
# #             "start":start,
# #             "path":paths
# #         }
# #         result["routes"].append(route)
# #     return result

# # # %%
# # map = []
# # width = t.map_width
# # height = t.map_height
# # lenght_map = (2 * t.map_width - 1) * (t.map_height // 2) + (t.map_height %
# #                                                             2) * t.map_width
# # deployment_points = []
# # entrances_points = []
# # exit_points = []
# # waypoints = []

# # for n in range(lenght_map):
# #     x, y = t.idx2xy(n)
# #     m = t.map[y][x]
# #     map.append(m)
# #     if m == 3:
# #         deployment_points.append((x, y))
# #     elif m == 4:
# #         entrances_points.append((x, y))
# #         waypoints.append((x, y))
# #     elif m == 5:
# #         exit_points.append((x, y))
# #         waypoints.append((x, y))
# #     elif m == 6:
# #         waypoints.append((x, y))

# # # # %%
# # # sequence = get_all_route_sequence(waypoints,
# # #                                   entrances_points,
# # #                                   exit_points,
# # #                                   is_drone=True)
# # # print(len(sequence))
# # # # %%
# # # with open('resource/aerial_routing.csv', 'w') as f:
# # #     for ds, rs in sequence:
# # #         ds = [str(d) for d in ds]
# # #         line = f'{rs[0][0]},{rs[0][1]},' + ','.join(ds)
# # #         f.write(line + '\n')
# # # # %%

# # # %%
# # sequence = get_all_route_sequence(waypoints,
# #                                   entrances_points,
# #                                   exit_points,
# #                                   is_drone=False)
# # result = convert_structure(sequence)
# # with open('resource/grounding_path.json','w') as f:
# #     json.dump(result,f,indent=4)

# # # print(len(sequence))
# # # for path in sequence:
# # #     print(path)
# # #     print("\n")
# # # %%
# # # with open('resource/ground_routing.csv', 'w') as f:
# # #     for ds, rs in sequence:
# # #         ds = [str(d) for d in ds]
# # #         line = f'{rs[0][0]},{rs[0][1]},' + ','.join(ds)
# # #         f.write(line + '\n')
# # # # %%
# # # # 创建一个空列表用于存储所有的route
# # # routes_list = []

# # # # 遍历sequence中的每个元素
# # # for ds, rs in sequence:
# # #     # 转换ds中的每个元素为字符串
# # #     ds = [str(d) for d in ds]
# # #     # 查找是否已有相同的rs在routes_list中
# # #     found = False
# # #     for route in routes_list:
# # #         if route['start'] == rs:
# # #             route['path'].append(ds)
# # #             found = True
# # #             break
# # #     # 如果没有找到相同的rs，创建一个新的route
# # #     if not found:
# # #         routes_list.append({'start': (rs[0][0],rs[0][1]), 'path': [ds]})

# # # # 创建最终的字典
# # # routing_dict = {"routes": routes_list}

# # # # 可以选择将字典保存到文件，或进行其他处理
# # # # 这里打印字典查看内容
# # # print(routing_dict)
# import csv
# import json
# from collections import defaultdict

# # 移动方向函数：1 - 上, 2 - 右上, 3 - 右, 4 - 右下, 5 - 下, 6 - 左下, 7 - 左, 8 - 左上
# def _move(x, y, direction):
#     if direction == 1:
#         y -= 1  # 上
#     elif direction == 2:
#         x += 1  # 右上
#         y -= 1
#     elif direction == 3:
#         x += 1  # 右
#     elif direction == 4:
#         x += 1  # 右下
#         y += 1
#     elif direction == 5:
#         y += 1  # 下
#     elif direction == 6:
#         x -= 1  # 左下
#         y += 1
#     elif direction == 7:
#         x -= 1  # 左
#     elif direction == 8:
#         x -= 1
#         y -= 1  # 左上
#     return (x, y)

# # 解析地图CSV文件
# def parse_map(filepath):
#     map_data = []
#     with open(filepath, 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             map_data.append([int(cell) for cell in row])
#     return map_data

# # 找到地图中的关键点：起始点、路径点和撤离点
# def find_points(map_data):
#     start_points = []
#     path_points = []
#     evac_points = []
    
#     for y in range(len(map_data)):
#         for x in range(len(map_data[0])):
#             if map_data[y][x] == 4:
#                 start_points.append((x, y))  # 出发点
#             elif map_data[y][x] == 6:
#                 path_points.append((x, y))  # 路径点
#             elif map_data[y][x] == 5:
#                 evac_points.append((x, y))  # 撤离点
    
#     return start_points, path_points, evac_points

# # 检查是否是有效的移动
# def is_valid_move(x, y, map_data):
#     if 0 <= x < len(map_data[0]) and 0 <= y < len(map_data):
#         return map_data[y][x] in [0, 6, 5]  # 只有街道、路径点或撤离点是有效的移动
#     return False

# # 生成两点之间的路径，保证路径尽量接近直线
# def generate_path_between_points(start, end):
#     path = []
#     current_pos = start
#     while current_pos != end:
#         x_diff = end[0] - current_pos[0]
#         y_diff = end[1] - current_pos[1]
        
#         direction = None
#         if abs(x_diff) >= abs(y_diff):
#             direction = 3 if x_diff > 0 else 7
#         else:
#             direction = 5 if y_diff > 0 else 1
        
#         next_pos = _move(current_pos[0], current_pos[1], direction)
#         path.append(direction)
#         current_pos = next_pos
    
#     return path

# # 生成从起始点到路径点，再到撤离点的路径
# def generate_straight_paths(map_data, start_points, path_points, evac_points):
#     routes = []
    
#     for start in start_points:
#         for path_point in path_points:
#             for evac_point in evac_points:
#                 # 生成从起始点到路径点，再到撤离点的路径
#                 route1 = generate_path_between_points(start, path_point)
#                 route2 = generate_path_between_points(path_point, evac_point)
                
#                 if route1 and route2:
#                     final_route = route1 + route2
#                     routes.append((final_route, start))
    
#     return routes

# # 转换路径结构为指定格式
# def convert_structure(data):
#     path_dict = defaultdict(list)
#     for item in data:
#         main_data = item[0]
#         start = item[1]
#         path_dict[start].append(main_data)
#     result = {"routes": []}
#     for start, paths in path_dict.items():
#         route = {
#             "start": list(start),
#             "path": paths
#         }
#         result["routes"].append(route)
#     return result

# # 生成所有路线的主函数
# def generate_routes(filepath):
#     map_data = parse_map(filepath)
#     start_points, path_points, evac_points = find_points(map_data)
    
#     all_routes = generate_straight_paths(map_data, start_points, path_points, evac_points)
    
#     result = convert_structure(all_routes)
#     return result

# # 地图CSV文件的路径
# map_file_path = 'resource/map.csv'

# # 生成路径字典
# routes = generate_routes(map_file_path)

# # 将结果保存到JSON文件
# output_file_path = 'resource/11-grounding_path.json'
# with open(output_file_path, 'w', encoding='utf-8') as json_file:
#     json.dump(routes, json_file, ensure_ascii=False, indent=4)

# print(f"路径已保存到: {output_file_path}")
import json
import copy

def _move(x, y, direction):
    if direction == 1:
        y -= 1  # 上
    elif direction == 2:
        x += 1  # 右上
        y -= 1
    elif direction == 3:
        x += 1  # 右
    elif direction == 4:
        x += 1  # 右下
        y += 1
    elif direction == 5:
        y += 1  # 下
    elif direction == 6:
        x -= 1  # 左下
        y += 1
    elif direction == 7:
        x -= 1  # 左
    elif direction == 8:
        x -= 1
        y -= 1  # 左上
    return (x, y)

def is_valid_position(x, y, map_data):
    if 0 <= x < len(map_data[0]) and 0 <= y < len(map_data):
        return map_data[y][x] in [0, 5, 6]  # 允许通过的位置：0（道路），5（撤离点），6（路径点）
    return False

def detect_invalid_path(path, start, N=10, M=4):
    current_pos = start
    visited_positions = []
    visited_positions.append(current_pos)

    for i, direction in enumerate(path):
        next_pos = _move(current_pos[0], current_pos[1], direction)

        # 判断绕圈逻辑：与N步之前的位置比较
        if i >= N:
            if manhattan_distance(next_pos, visited_positions[i - N]) < M:
                return True

        visited_positions.append(next_pos)
        current_pos = next_pos

    return False

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def filter_paths(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    cnt = 0
    filtered_routes = []

    for route in data['routes']:
        start = tuple(route['start'])
        valid_paths = []
        
        for path in route['path']:
            # 使用路径的深拷贝进行处理，避免原始数据被修改
            path_copy = copy.deepcopy(path)
            if not detect_invalid_path(path_copy, start):
                valid_paths.append(path)
                cnt+=1

        if valid_paths:
            filtered_routes.append({
                'start': start,
                'path': valid_paths
            })

    filtered_data = {'routes': filtered_routes}
    print(f"合格路径数量: {cnt}")
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

    print(f"过滤后的路径已保存到: {output_filepath}")


# 输入和输出文件路径
input_filepath = 'resource/grounding_path-backup.json'
output_filepath = 'resource/grounding_path.json'

filter_paths(input_filepath, output_filepath)
