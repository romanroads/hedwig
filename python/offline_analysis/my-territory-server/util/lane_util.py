# !/usr/bin/python
# coding=utf-8

import math

import numpy as np
from shapely.geometry import Point
from config import default_config
from entity.lane import Lane
from entity.line import Line


# 道路设置
default_lanes = {}

# 画面像素和实际单位的比例
pixel_ratio = None

# 排除左边的x左边值
exclude_left_x = None

# 轿车的宽度差范围（统计后的宽度范围-标准的宽度范围）
car_height_diff_range = [0, 0]

# 货车的宽度差范围（统计后的宽度范围-标准的宽度范围）
truck_height_diff_range = [0, 0]


def set_background(background):
    global default_lanes
    global pixel_ratio
    global exclude_left_x
    global car_height_diff_range
    global truck_height_diff_range

    pixel_ratio = background.pixel_ratio
    exclude_left_x = background.exclude_left_x
    statistics_car_height_range = background.statistics_car_height_range
    statistics_truck_height_range = background.statistics_truck_height_range

    # 之前统计已经减去了default_config.car_bb_fixed的边框修正值，这次再加回来
    if statistics_car_height_range:
        car_height_diff_range = np.array(default_config.car_height_range) - np.array(statistics_car_height_range)

    if statistics_truck_height_range:
        truck_height_diff_range = np.array(default_config.truck_height_range) - np.array(statistics_truck_height_range)

    print(car_height_diff_range)
    print(truck_height_diff_range)

    # 车道线数量
    line_num = len(background.default)
    # 车道数量
    lane_num = line_num - 2
    # 计算一半车道数
    half_num = int(lane_num / 2)
    # 生成车道，多加一股道，中间值为绿化带
    for i in range(1, lane_num + 1):
        # 计算车道ID
        if i <= half_num:
            lane_id = i  # 车道ID
            up_or_down = 1  # 上车道或下车道
        else:
            lane_id = i + 1  # 车道ID
            up_or_down = 2  # 上车道或下车道

        # 计算是否靠近中间绿化带
        is_near_mid = False
        if i == half_num or i == (half_num + 1):
            is_near_mid = True

        # 上下车道线
        top_lane = background.default.get(lane_id)
        bottom_lane = background.default.get(lane_id + 1)

        # 计算倾斜
        angle, cross_p = cale_angle(top_lane)

        lane = Lane(top_lane, bottom_lane, up_or_down, is_near_mid, angle, cross_p, pixel_ratio, exclude_left_x)

        default_lanes.update({lane_id: lane})


def get_lane(p):
    """
    获取车道信息
    :param p: 车辆坐标
    :return: 车道信息
    """

    # 获取车道信息
    for lane_id, lane in dict(default_lanes).items():
        # 判断车道矩形是否包含当前坐标
        if lane.lane_polygon.contains(p):
            return lane_id, lane
    return None, None


def get_fix_lane(p):
    """
    获取校正后车道信息
    :param p: 车辆坐标
    :return: 车道信息
    """

    # 获取车道信息
    for lane_id, lane in dict(default_lanes).items():
        # 判断车道矩形是否包含当前坐标
        if lane.fix_lane_polygon.contains(p):
            return lane_id, lane
    return None, None


def cale_angle(line):
    """
    计算倾斜角度
    :param line: 车道线
    :return: 倾斜角度
    """

    l1_p1 = Point(0, line[0])  # 标准车道的第一条车道线左坐标
    l1_p2 = Point(default_config.width, line[0])  # 标准车道的第一条车道线右坐标
    l1 = Line(l1_p1, l1_p2)  # 标准的车道线

    l2_p1 = Point(0, line[0])  # 倾斜车道线左坐标
    l2_p2 = Point(default_config.width, line[1])  # 倾斜车道线右坐标
    l2 = Line(l2_p1, l2_p2)  # 倾斜的车道线

    angle = line_angle(l1, l2)  # 倾斜角度
    # 没有倾斜角度
    if angle == 0:
        return angle, None

    cross_p = get_cross_point(l1, l2)  # 两条线的交点

    return angle, cross_p


def is_in_register(lane, p):
    """
    判断是否在注册区
    :param lane: 车道信息
    :param p: 车辆坐标
    :return: 是否标识
    """

    # 判断注册区是否包含当前车辆的坐标
    if lane.register_polygon.contains(p):
        return True

    return False


def is_in_destroy(lane, p):
    """
    判断是否在销毁区区
    :param lane: 车道信息
    :param p: 车辆坐标
    :return: 是否标识
    """

    # 判断销毁区是否包含当前车辆的坐标
    if lane.destroy_polygon.contains(p):
        return True

    return False


def line_angle(l1, l2):
    """
    两条线之间的夹角角度（180°）
    :param l1:线1
    :param l2:线2
    :return:夹角角度
    """

    dx1 = l1.p2.x - l1.p1.x
    dy1 = l1.p2.y - l1.p1.y
    dx2 = l2.p2.x - l2.p1.x
    dy2 = l2.p2.y - l2.p1.y
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180 / math.pi
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180 / math.pi
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle

    if l2.p2.y > l1.p2.y:
        included_angle = 360 - included_angle
    return included_angle


def rotation_point(center, point, angle):
    """
    坐标旋转
    :param center: 旋转顶点
    :param point: 坐标
    :param angle: 角度
    :param off_set: 偏移量
    :return: 旋转后的坐标
    """

    if angle == 0:
        return point

    radian = math.radians(angle)

    x = (point.x - center.x) * math.cos(radian) - (point.y - center.y) * math.sin(radian) + center.x
    y = (point.x - center.x) * math.sin(radian) + (point.y - center.y) * math.cos(radian) + center.y

    return Point(x, y)


def get_line_para(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y


def get_cross_point(l1, l2):
    """
    获取两条线的焦点
    :param l1: 线1
    :param l2: 线2
    :return: 焦点坐标
    """
    get_line_para(l1)
    get_line_para(l2)
    d = l1.a * l2.b - l2.a * l1.b

    x = (l1.b * l2.c - l2.b * l1.c) * 1.0 / d
    y = (l1.c * l2.a - l2.c * l1.a) * 1.0 / d
    return Point(x, y)


def lane_handle(old_lane_arr):
    """
    处理车道ID浮动
    :param old_lane_arr: 原始车道ID集合
    :return:
    """

    new_lane_arr = []
    lane_size = len(old_lane_arr)
    lane_windows = 31
    index = 0
    for lane_id in old_lane_arr:
        start_index = index
        end_index = start_index + lane_windows
        # 如果超出数组界限，则待计算的数组为最后lane_windows大小的数组
        if end_index >= lane_size:
            start_index = lane_size - lane_windows
            end_index = lane_size
        # 获取计算的窗口数组里出现次数最多的元素
        cur_lane_id = np.argmax(np.bincount(old_lane_arr[start_index: end_index]))
        new_lane_arr.append(cur_lane_id)

        index = index + 1

    return new_lane_arr
