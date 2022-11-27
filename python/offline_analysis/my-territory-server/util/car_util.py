# !/usr/bin/python
# coding=utf-8

from bisect import bisect_left

from shapely.geometry import Polygon

# 两个多边形的判断重叠的比例阈值
IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP = 0.01


def __cal_inter_area(poly1, poly2):
    """
    任意两个多边形的交集面积的计算
    :param poly1: 多边形1
    :param poly2: 多边形2
    :return: 两个多边形的交集面积
    """

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


def __cal_union_area(poly1, poly2):
    """
    任意两个多边形的并集面积的计算
    :param poly1: 多边形1
    :param poly2: 多边形2
    :return: 两个多边形的并集面积
    """

    union_area = poly1.union(poly2).area
    return union_area


def is_duplicate(car_data1, car_data2):
    """
    判断任意两个车是否重叠
    :param car_data1: 其他车
    :param car_data2: 当前车
    :return: 是否重叠
    """

    fractional_top_left_x = car_data1.fractional_center_x - 0.5 * car_data1.width
    fractional_top_left_y = car_data1.fractional_center_y - 0.5 * car_data1.height
    fractional_bottom_right_x = car_data1.fractional_center_x + 0.5 * car_data1.width
    fractional_bottom_right_y = car_data1.fractional_center_y + 0.5 * car_data1.height

    # 构建车辆的多边形
    data1 = [(fractional_top_left_x, fractional_top_left_y),
             (fractional_top_left_x, fractional_bottom_right_y),
             (fractional_bottom_right_x, fractional_bottom_right_y),
             (fractional_bottom_right_x, fractional_top_left_y)]

    fractional_top_left_x = car_data2.fractional_center_x - 0.5 * car_data2.width
    fractional_top_left_y = car_data2.fractional_center_y - 0.5 * car_data2.height
    fractional_bottom_right_x = car_data2.fractional_center_x + 0.5 * car_data2.width
    fractional_bottom_right_y = car_data2.fractional_center_y + 0.5 * car_data2.height

    data2 = [(fractional_top_left_x, fractional_top_left_y),
             (fractional_top_left_x, fractional_bottom_right_y),
             (fractional_bottom_right_x, fractional_bottom_right_y),
             (fractional_bottom_right_x, fractional_top_left_y)]

    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    inter_area = __cal_inter_area(poly1, poly2)  # 计算相交面积
    # print("两个图形交集面积：", inter_area)

    union_area = __cal_union_area(poly1, poly2)  # 计算合并面积
    # print("两个图形并集面积：", union_area)

    iou = inter_area / union_area  # 计算交并比
    # print("两个图形交并比：", iou)

    is_iou_too_large = iou > IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP  # 大于设置的比例（当前为50%），则为重叠
    # print("两个图形是否重叠：", is_iou_too_large)

    return is_iou_too_large


def is_nearest_duplicate(nearest_car_data, car_data):
    """
    判断任意两个车是否重叠
    :param nearest_car_data: 相邻车
    :param car_data: 当前车
    :return: 是否重叠
    """

    # 构建车辆的多边形
    data1 = [(nearest_car_data.fractional_top_left_x, nearest_car_data.fractional_top_left_y),
             (nearest_car_data.fractional_top_left_x, nearest_car_data.fractional_bottom_right_y),
             (nearest_car_data.fractional_bottom_right_x, nearest_car_data.fractional_bottom_right_y),
             (nearest_car_data.fractional_bottom_right_x, nearest_car_data.fractional_top_left_y)]

    # 更改当前车的y轴坐标，与相邻的车判断重合度
    car_data_bottom_right_y = car_data.fractional_bottom_right_y \
                              - (car_data.fractional_top_left_y - nearest_car_data.fractional_top_left_y)
    data2 = [(car_data.fractional_top_left_x, nearest_car_data.fractional_top_left_y),
             (car_data.fractional_top_left_x, car_data_bottom_right_y),
             (car_data.fractional_bottom_right_x, car_data_bottom_right_y),
             (car_data.fractional_bottom_right_x, nearest_car_data.fractional_top_left_y)]

    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    inter_area = __cal_inter_area(poly1, poly2)  # 计算相交面积
    # print("两个图形交集面积：", inter_area)

    union_area = __cal_union_area(poly1, poly2)  # 计算合并面积
    # print("两个图形并集面积：", union_area)

    iou = inter_area / union_area  # 计算交并比
    # print("两个图形交并比：", iou)

    is_iou_too_large = iou > IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP  # 大于设置的比例（当前为50%），则为重叠
    # print("两个图形是否重叠：", is_iou_too_large)

    return is_iou_too_large


def get_nearest(car_tracks, car_data):
    """
    获取最近的车辆轨迹
    :param car_tracks: 车辆轨迹数据集合
    :param car_data: 当前车辆数据
    :return: 最近的车辆轨迹
    """

    if car_data.fractional_center_x < car_tracks[0].fractional_center_x:
        return car_tracks[0]

    if car_data.fractional_center_x > car_tracks[-1].fractional_center_x:
        return car_tracks[-1]

    x_list = [track.fractional_center_x for track in car_tracks]
    pos = bisect_left(x_list, car_data.fractional_center_x)

    if pos == 0:
        return car_tracks[0]
    if pos == len(car_tracks):
        return car_tracks[-1]
    before = car_tracks[pos - 1]
    after = car_tracks[pos]
    if after.fractional_center_x - car_data.fractional_center_x \
            < car_data.fractional_center_x - before.fractional_center_x:
        return after
    else:
        return before


def get_adjacent_car_id(road_tracks, car_data):
    """
    获取相邻车道的前中后车辆ID
    :param road_tracks: 相邻车道的车辆轨迹集合
    :param car_data: 当前车辆
    :return: 相邻车道的前中后车辆ID
    """

    preceding_id = 0  # 前车ID
    alongside_id = 0  # 中车ID
    following_id = 0  # 后车ID

    # 获取最近的车辆轨迹数据
    nearest_car_data = get_nearest(road_tracks, car_data)
    nearest_car_index = road_tracks.index(nearest_car_data)
    nearest_is_before = False
    # 判断重合度，是否为平行车辆
    if is_nearest_duplicate(nearest_car_data, car_data):
        alongside_id = nearest_car_data.id
    # 判断最近的车辆是否在数组的前面
    elif nearest_car_data.fractional_center_x - car_data.fractional_center_x < 0:
        nearest_is_before = True
    # 判断是上车道还是下车道，1：上车道  2：下车道
    if car_data.up_or_down == 1:
        # 如果有平行车辆或者最近的车辆在当前车辆数据之后，则计算相邻车道的前车ID
        if alongside_id > 0:
            # 获取相邻车道的前车ID
            if nearest_car_index > 0:
                preceding_id = road_tracks[nearest_car_index - 1].id
            # 获取相邻车道的后车ID
            if nearest_car_index < len(road_tracks) - 1:
                following_id = road_tracks[nearest_car_index + 1].id
        else:
            if not nearest_is_before:
                following_id = nearest_car_data.id
                # 获取相邻车道的前车ID
                if nearest_car_index > 0:
                    preceding_id = road_tracks[nearest_car_index - 1].id
            else:
                preceding_id = nearest_car_data.id
                # 获取相邻车道的后车ID
                if nearest_car_index < len(road_tracks) - 1:
                    following_id = road_tracks[nearest_car_index + 1].id
    else:
        # 如果有平行车辆或者最近的车辆在当前车辆数据之后，则计算相邻车道的后车ID
        if alongside_id > 0:
            # 获取相邻车道的后车ID
            if nearest_car_index > 0:
                following_id = road_tracks[nearest_car_index - 1].id
            # 获取相邻车道的前车ID
            if nearest_car_index < len(road_tracks) - 1:
                preceding_id = road_tracks[nearest_car_index + 1].id
        else:
            if not nearest_is_before:
                preceding_id = nearest_car_data.id
                # 获取相邻车道的后车ID
                if nearest_car_index > 0:
                    following_id = road_tracks[nearest_car_index - 1].id
            else:
                following_id = nearest_car_data.id
                # 获取相邻车道的前车ID
                if nearest_car_index < len(road_tracks) - 1:
                    preceding_id = road_tracks[nearest_car_index + 1].id
    return preceding_id, alongside_id, following_id



