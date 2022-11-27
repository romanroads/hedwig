# !/usr/bin/python
# coding=utf-8
import codecs
import csv
import os

import numpy as np
import pandas
from scipy import interpolate
from shapely.geometry import Point
from scipy.signal import savgol_filter
from config import default_config
from entity.line import Line
from input import udi_a
from input.udi_a import InputEntity
from output import tracks, tracks_extra
from output.tracks import Track
from util import lane_util, car_util, tracks_mate_util
import statsmodels.api as sm

lowess = sm.nonparametric.lowess

# 轨迹集合
tracks_dict = {}

# 最大帧数
max_frame = 0

# ID对应字典
id_dict = {}


def pre_filter(chunk_combined, isLast):
    """
    预处理
    :param chunk_combined: 合并后的chunk
    """
    global tracks_dict

    df = chunk_combined

    # 最后一批数据时，取最大帧数
    if isLast:
        global max_frame
        max_frame = max(df[udi_a.FRAME_ID].values)
    print("max_frame:{}".format(max_frame))

    # 筛选后的数据集合
    filter_data = []
    last_id = 0
    last_center_x = 0
    for index, row in df.iterrows():
        #if row[udi_a.ID] != 102:
        #   continue
        car_id = row[udi_a.ID]
        frame_id = row[udi_a.FRAME_ID]
        fractional_top_left_x = row[udi_a.FRACTIONAL_TOP_LEFT_X] * default_config.width / lane_util.pixel_ratio
        fractional_top_left_y = row[udi_a.FRACTIONAL_TOP_LEFT_Y] * default_config.height / lane_util.pixel_ratio
        fractional_bottom_right_x = row[udi_a.FRACTIONAL_BOTTOM_RIGHT_X] * default_config.width / lane_util.pixel_ratio
        fractional_bottom_right_y = row[udi_a.FRACTIONAL_BOTTOM_RIGHT_Y] * default_config.height / lane_util.pixel_ratio
        fractional_center_x = row[udi_a.FRACTIONAL_CENTER_X] * default_config.width / lane_util.pixel_ratio
        fractional_center_y = row[udi_a.FRACTIONAL_CENTER_Y] * default_config.height / lane_util.pixel_ratio
        top_left_p = Point(fractional_top_left_x, fractional_top_left_y)
        bottom_right_p = Point(fractional_bottom_right_x, fractional_bottom_right_y)
        center_p = Point(fractional_center_x, fractional_center_y)
        tag = row[udi_a.TAG]
        if tag:
            tracks_dict.update({car_id: 1})
            continue
        else:
            if tracks_dict.get(car_id) != 0:
                tracks_dict.update({car_id: 3})

        # 排除x速度为0的数据
        if row[udi_a.ID] == last_id and fractional_center_x == last_center_x:
            continue
        else:
            last_id = row[udi_a.ID]
            last_center_x = fractional_center_x

        # 获取当前车辆所在车道，为None时表示不在车道范围内
        lane_id, lane = lane_util.get_lane(center_p)
        if not lane:
            # print("当前车辆不在车道范围内，数据丢弃")
            # print("当前数据[id:{},frame_id:{},center_p:{}]".format(id, frame_id, center_p))
            # print("旋转前坐标：[id:{},frame_id:{},center_p:{}]".format(id, frame_id, center_p_old))
            continue

        # 判断车辆是否为探头车辆
        if row[udi_a.FRACTIONAL_CENTER_X] < 0.03 or row[udi_a.FRACTIONAL_CENTER_X] > 0.97:
            continue

        # 将坐标进行水平映射
        top_left_p = lane_util.rotation_point(lane.cross_p, top_left_p, lane.angle)
        bottom_right_p = lane_util.rotation_point(lane.cross_p, bottom_right_p, lane.angle)
        center_p = lane_util.rotation_point(lane.cross_p, center_p, lane.angle)

        row[udi_a.FRACTIONAL_TOP_LEFT_X] = top_left_p.x
        row[udi_a.FRACTIONAL_TOP_LEFT_Y] = top_left_p.y
        row[udi_a.FRACTIONAL_BOTTOM_RIGHT_X] = bottom_right_p.x
        row[udi_a.FRACTIONAL_BOTTOM_RIGHT_Y] = bottom_right_p.y
        row[udi_a.FRACTIONAL_CENTER_X] = center_p.x
        row[udi_a.FRACTIONAL_CENTER_Y] = center_p.y
        row[udi_a.LANE_ID] = lane_id

        # 加入筛选后的集合
        filter_data.append(row)

    return filter_data


def handle_step_1(data):
    """
    第一步数据处理
    :return:
    """

    total_need_count = 0

    print("-----开始执行handle_step_1()")
    global tracks_dict

    df = pandas.DataFrame(data)

    car_grouped = df.groupby(udi_a.ID)

    # 所有车辆的轨迹数据缓存集合（按车辆ID分组）
    cars_cache = {}

    # 所有车辆的轨迹数据缓存集合（按祯ID分组）
    frames_cache = {}

    for car_id, car_group_data in car_grouped:
        # 获取车辆各坐标数据集合
        old_frame_id_arr = list(car_group_data[udi_a.FRAME_ID].values)
        old_fractional_center_x_arr = list(car_group_data[udi_a.FRACTIONAL_CENTER_X].values)
        old_fractional_center_y_arr = list(car_group_data[udi_a.FRACTIONAL_CENTER_Y].values)
        old_fractional_top_left_x_arr = list(car_group_data[udi_a.FRACTIONAL_TOP_LEFT_X].values)
        old_fractional_top_left_y_arr = list(car_group_data[udi_a.FRACTIONAL_TOP_LEFT_Y].values)
        old_fractional_bottom_right_x_arr = list(car_group_data[udi_a.FRACTIONAL_BOTTOM_RIGHT_X].values)
        old_fractional_bottom_right_y_arr = list(car_group_data[udi_a.FRACTIONAL_BOTTOM_RIGHT_Y].values)

        # 补全缺失的祯
        frame_id_arr = np.arange(min(old_frame_id_arr), max(old_frame_id_arr) + 1, 1)

        # 排除小于100祯的车，可能是脏数据
        if len(frame_id_arr) < 100:
            continue

        """
        kind方法：
        nearest、zero、slinear、quadratic、cubic
        实现函数func
        """
        # 利用插值法补全数据
        func = interpolate.interp1d(old_frame_id_arr, old_fractional_center_x_arr, kind='nearest')
        kind_center_x_arr = func(frame_id_arr)

        func = interpolate.interp1d(old_frame_id_arr, old_fractional_center_y_arr, kind='nearest')
        kind_center_y_arr = func(frame_id_arr)

        func = interpolate.interp1d(old_frame_id_arr, old_fractional_top_left_x_arr, kind='nearest')
        kind_top_left_x_arr = func(frame_id_arr)

        func = interpolate.interp1d(old_frame_id_arr, old_fractional_top_left_y_arr, kind='nearest')
        kind_top_left_y_arr = func(frame_id_arr)

        func = interpolate.interp1d(old_frame_id_arr, old_fractional_bottom_right_x_arr, kind='nearest')
        kind_bottom_right_x_arr = func(frame_id_arr)

        func = interpolate.interp1d(old_frame_id_arr, old_fractional_bottom_right_y_arr, kind='nearest')
        kind_bottom_right_y_arr = func(frame_id_arr)

        # 计算车辆宽度平均值,高度平均值
        width = np.mean(kind_bottom_right_x_arr) - np.mean(kind_top_left_x_arr) - default_config.car_bb_fixed
        height = np.mean(kind_bottom_right_y_arr) - np.mean(kind_top_left_y_arr) - default_config.car_bb_fixed

        # 判断是轿车还是货车
        if width <= 8:
            # 如果小于最小值
            if height < default_config.car_height_range[0]:
                # 差值大于0表示最小宽度超出标准最小宽度
                if lane_util.car_height_diff_range[0] > 0:
                    new_height = height + lane_util.car_height_diff_range[0]
                    # 如果新的宽度超过标准最大宽度，则宽度=标准最小宽度 + 一半的超出宽度
                    if new_height > default_config.car_height_range[1]:
                        new_height = height + lane_util.car_height_diff_range[0] / 3

                    # 计算缩放比例
                    fix_rate = abs(lane_util.car_height_diff_range[0]) / height
                    # 增加height
                    height = new_height
                    # 增加width
                    width = width * (1 + fix_rate)
            # 如果大于最大值
            elif height > default_config.car_height_range[1]:
                # 差值小于0表示最大宽度超出标准最大宽度
                if lane_util.car_height_diff_range[1] < 0:
                    new_height = height + lane_util.car_height_diff_range[1]
                    # 如果新的宽度超过标准最小宽度，则宽度=标准最大宽度 - 一半的超出宽度
                    if new_height < default_config.car_height_range[0]:
                        new_height = height + lane_util.car_height_diff_range[1] / 3

                    # 计算缩放比例
                    fix_rate = abs(lane_util.car_height_diff_range[1]) / height
                    # 减少height
                    height = new_height
                    # 减少width
                    width = width * (1 - fix_rate)
        else:
            # 如果小于最小值
            if height < default_config.truck_height_range[0]:
                # 差值大于0表示最小宽度超出标准最小宽度
                if lane_util.truck_height_diff_range[0] > 0:
                    new_height = height + lane_util.truck_height_diff_range[0]
                    # 如果新的宽度超过标准最大宽度，则宽度=标准最小宽度 + 一半的超出宽度
                    if new_height > default_config.truck_height_range[1]:
                        new_height = height + lane_util.truck_height_diff_range[0]/3

                    # 计算缩放比例
                    fix_rate = abs(lane_util.truck_height_diff_range[0]) / height
                    # 增加height
                    height = new_height
                    # 增加width
                    width = width * (1 + fix_rate)
            # 如果大于最大值
            elif height > default_config.truck_height_range[1]:
                # 差值小于0表示最大宽度超出标准最大宽度
                if lane_util.truck_height_diff_range[1] < 0:
                    new_height = height + lane_util.truck_height_diff_range[1]
                    # 如果新的宽度超过标准最小宽度，则宽度=标准最大宽度 - 一半的超出宽度
                    if new_height < default_config.truck_height_range[0]:
                        new_height = height + lane_util.truck_height_diff_range[1] / 3

                    # 计算缩放比例
                    fix_rate = abs(lane_util.truck_height_diff_range[1]) / height
                    # 减少height
                    height = new_height
                    # 减少width
                    width = width * (1 - fix_rate)


        # x坐标平滑处理
        sm_x = lowess(kind_center_x_arr, np.arange(0, len(kind_center_x_arr), 1), frac=0.1)
        filter_x_arr = sm_x[:, 1]

        # x速度计算
        old_x_v_arr = np.diff(filter_x_arr) * default_config.fps
        if len(old_x_v_arr) > 1:
            fisrt_x_velocity = old_x_v_arr[0] - (old_x_v_arr[1] - old_x_v_arr[0])
        else:
            fisrt_x_velocity = 0
        old_x_v_arr = np.insert(old_x_v_arr, 0, fisrt_x_velocity)  # 补充第一个x速度

        # x速度平滑处理
        sm_x_v = lowess(old_x_v_arr, np.arange(0, len(old_x_v_arr), 1), frac=0.1)
        filter_x_v_arr = sm_x_v[:, 1]

        # x加速度计算
        old_x_a_v_arr = np.diff(filter_x_v_arr) * default_config.fps
        if len(old_x_a_v_arr) > 1:
            fisrt_x_a_velocity = old_x_a_v_arr[0] - (old_x_a_v_arr[1] - old_x_a_v_arr[0])
        else:
            fisrt_x_a_velocity = 0
        old_x_a_v_arr = np.insert(old_x_a_v_arr, 0, fisrt_x_a_velocity)  # 补充第一个x加速度

        # x加速度平滑处理
        sm_x_a_v = lowess(old_x_a_v_arr, np.arange(0, len(old_x_a_v_arr), 1), frac=0.1)
        filter_x_a_v_arr = sm_x_a_v[:, 1]

        # 重新计算x速度
        filter_x_a_v_arr = filter_x_a_v_arr / default_config.fps

        fractional_center_x_arr = [filter_x_arr[0]]
        x_velocity_arr = [filter_x_v_arr[0]]
        x_a_velocity_arr = [filter_x_a_v_arr[0]]
        index = 0
        is_out_xa = False
        for x_a_v in filter_x_a_v_arr:
            # 判断x加速度是否超过范围
            if x_a_v * default_config.fps > default_config.xa_range[0] or x_a_v * default_config.fps < default_config.xa_range[1]:
                is_out_xa = True
                break
            if index > 0:
                cur_x_v = x_velocity_arr[-1] + x_a_v
                x_velocity_arr.append(cur_x_v)
                cur_x = fractional_center_x_arr[-1] + cur_x_v / default_config.fps
                fractional_center_x_arr.append(cur_x)
                x_a_velocity_arr.append(x_a_v)
            index += 1

        x_a_velocity_arr = np.array(x_a_velocity_arr) * default_config.fps

        # 异常x加速度校验
        if is_out_xa:
            print("当前车辆ID:{},x加速度超出范围".format(car_id))
            continue

        # y坐标平滑处理
        sm_y = lowess(kind_center_y_arr, np.arange(0, len(kind_center_y_arr), 1), frac=0.1)
        filter_y_arr = sm_y[:, 1]

        # y速度计算
        old_y_v_arr = np.diff(filter_y_arr) * default_config.fps
        if len(old_y_v_arr) > 1:
            fisrt_y_velocity = old_y_v_arr[0] - (old_y_v_arr[1] - old_y_v_arr[0])
        else:
            fisrt_y_velocity = 0
        old_y_v_arr = np.insert(old_y_v_arr, 0, fisrt_y_velocity)  # 补充第一个y速度

        # y速度平滑处理
        sm_y_v = lowess(old_y_v_arr, np.arange(0, len(old_y_v_arr), 1), frac=0.1)
        filter_y_v_arr = sm_y_v[:, 1]

        # y加速度计算
        old_y_a_v_arr = np.diff(filter_y_v_arr) * default_config.fps
        if len(old_y_a_v_arr) > 1:
            fisrt_y_a_velocity = old_y_a_v_arr[0] - (old_y_a_v_arr[1] - old_y_a_v_arr[0])
        else:
            fisrt_y_a_velocity = 0
        old_y_a_v_arr = np.insert(old_y_a_v_arr, 0, fisrt_y_a_velocity)  # 补充第一个y加速度

        # y加速度平滑处理
        sm_y_a_v = lowess(old_y_a_v_arr, np.arange(0, len(old_y_a_v_arr), 1), frac=0.1)
        filter_y_a_v_arr = sm_y_a_v[:, 1]

        # 重新计算y速度
        filter_y_a_v_arr = filter_y_a_v_arr / default_config.fps

        fractional_center_y_arr = [filter_y_arr[0]]
        y_velocity_arr = [filter_y_v_arr[0]]
        y_a_velocity_arr = [filter_y_a_v_arr[0]]
        index = 0
        is_out_ya = False
        for y_a_v in filter_y_a_v_arr:
            # 判断y加速度是否超过范围
            if y_a_v * default_config.fps > default_config.ya_range[0] or y_a_v * default_config.fps < default_config.ya_range[1]:
                is_out_ya = True
                break
            if index > 0:
                cur_y_v = y_velocity_arr[-1] + y_a_v
                y_velocity_arr.append(cur_y_v)
                cur_y = fractional_center_y_arr[-1] + cur_y_v / default_config.fps
                fractional_center_y_arr.append(cur_y)
                y_a_velocity_arr.append(y_a_v)
            index += 1

        y_a_velocity_arr = np.array(y_a_velocity_arr) * default_config.fps

        # 异常y加速度校验
        if is_out_ya:
            print("当前车辆ID:{},y加速度超出范围".format(car_id))
            continue

        # 超出绿化带处理
        lane_id_arr = []
        max_out_offset = None
        for index, center_y in enumerate(fractional_center_y_arr):
            fractional_center_x = fractional_center_x_arr[index]
            fractional_center_y = fractional_center_y_arr[index]
            center_p = Point(fractional_center_x, fractional_center_y)
            lane_id, lane = lane_util.get_fix_lane(center_p)
            lane_id_arr.append(lane_id)
            if not lane_id:
                continue
            # 判断当前车道是否接近绿化带
            if lane.is_near_mid:
                # 对y坐标超出绿化带的特殊处理
                if lane.up_or_down == 1:
                    if (center_y + 0.5 * height) > lane.bottom_left_y:
                        out_offset = lane.bottom_left_y - (center_y + 0.5 * height)
                        if not max_out_offset or out_offset < max_out_offset:
                            max_out_offset = out_offset
                else:
                    if (center_y - 0.5 * height) < lane.top_left_y:
                        out_offset = lane.top_left_y - (center_y - 0.5 * height)
                        if not max_out_offset or out_offset > max_out_offset:
                            max_out_offset = out_offset
        if max_out_offset:
            fractional_center_y_arr = fractional_center_y_arr + max_out_offset

        # 重新计算车道ID、车辆边框
        fractional_top_left_x_arr = []
        fractional_top_left_y_arr = []
        fractional_bottom_right_x_arr = []
        fractional_bottom_right_y_arr = []
        for index, x in enumerate(fractional_center_x_arr):
            fractional_center_x = x
            fractional_center_y = fractional_center_y_arr[index]
            fractional_top_left_x = fractional_center_x - width * 0.5
            fractional_top_left_y = fractional_center_y - height * 0.5
            fractional_bottom_right_x = fractional_center_y + width * 0.5
            fractional_bottom_right_y = fractional_center_y + height * 0.5
            fractional_top_left_x_arr.append(fractional_top_left_x)
            fractional_top_left_y_arr.append(fractional_top_left_y)
            fractional_bottom_right_x_arr.append(fractional_bottom_right_x)
            fractional_bottom_right_y_arr.append(fractional_bottom_right_y)

        # 车辆ID
        if not bool(id_dict):
            id = 1
        elif car_id in id_dict.keys():
            id = id_dict.get(car_id)
        else:
            id = max(id_dict.values()) + 1
        print("当前车辆ID:{},新的ID:{}".format(car_id, id))

        # 是否经过注册区区
        is_in_register = False
        # 是否经过销毁区
        is_in_destroy = False
        # 筛选过后的车辆轨迹集合
        car_tracks = []
        error_num = 0  # 异常次数
        error_list = []  # 异常集合
        for index, frame_id in enumerate(frame_id_arr):
            # 如果是debug模式，则使用原数据的车辆ID，如果不是debug模式，则使用重新赋值的车辆ID
            if default_config.debug:
                id = car_id

            # 获取各项数值
            frame_id = frame_id_arr[index]
            fractional_top_left_x = fractional_top_left_x_arr[index]
            fractional_top_left_y = fractional_top_left_y_arr[index]
            fractional_bottom_right_x = fractional_bottom_right_x_arr[index]
            fractional_bottom_right_y = fractional_bottom_right_y_arr[index]
            fractional_center_x = fractional_center_x_arr[index]
            fractional_center_y = fractional_center_y_arr[index]
            x_velocity = x_velocity_arr[index]
            y_velocity = y_velocity_arr[index]
            x_a_velocity = x_a_velocity_arr[index]
            y_a_velocity = y_a_velocity_arr[index]

            # 排除左侧x设置的距离
            if fractional_center_x < lane_util.exclude_left_x:
                continue

            # 排除x速度为0的数据
            if x_velocity == 0:
                continue

            lane_id = lane_id_arr[index]
            if not lane_id:
                continue

            lane = lane_util.default_lanes.get(lane_id)

            center_p = Point(fractional_center_x, fractional_center_y)  # 车辆拟合后的中心点坐标

            # 判断异常变道
            if car_tracks and lane.up_or_down != car_tracks[-1].up_or_down:  # 如果上下车道不一致，则当前车辆轨迹为异常
                error_num = error_num + 1
                error_info = "上下车道变道异常---祯ID：{}，当前所在车道：{}，应在车道：{}".format(frame_id, lane.up_or_down,
                                                                          car_tracks[-1].up_or_down)
                error_list.append(error_info)
                continue

            # 判断当前位置是否异常，上车道正常x坐标递减，下车道正常x坐标递增
            if lane.up_or_down == 1 and car_tracks and fractional_center_x - car_tracks[-1].fractional_center_x > 0:
                error_num = error_num + 1
                error_info = "x坐标异常，x值应为递减---祯ID：{}，当前x值：{}，上一次x值：{}，差值：{}".format(frame_id, fractional_center_x,
                                                                                   pre_x,
                                                                                   fractional_center_x - pre_x)
                error_list.append(error_info)
                pre_x = fractional_center_x
                continue
            elif lane.up_or_down == 2 and car_tracks and fractional_center_x - car_tracks[-1].fractional_center_x < 0:
                error_num = error_num + 1
                error_info = "x坐标异常，x值应为递增---祯ID：{}，当前x值：{}，上一次x值：{}，差值：{}".format(frame_id, fractional_center_x,
                                                                                   pre_x,
                                                                                   fractional_center_x - pre_x)
                error_list.append(error_info)
                pre_x = fractional_center_x
                continue
            else:
                pre_x = fractional_center_x

            # 是否经过注册区
            if lane_util.is_in_register(lane, center_p):
                is_in_register = True

            # 是否经过销毁区
            if lane_util.is_in_destroy(lane, center_p):
                is_in_destroy = True

            # 如果车辆左边框的x坐标小于0，则丢弃该车辆数据
            if fractional_center_x - 0.5 * width < 0:
                continue

            # 设置实体属性
            input_data = InputEntity(
                id,
                car_id,
                frame_id,
                fractional_top_left_x,
                fractional_top_left_y,
                fractional_bottom_right_x,
                fractional_bottom_right_y,
                fractional_center_x,
                fractional_center_y,
                x_velocity,
                y_velocity,
                x_a_velocity,
                y_a_velocity,
                width,
                height,
                lane_id,
                lane.up_or_down,
                is_in_destroy
            )

            # 去除重复车辆，与当前帧的其他车辆比对
            is_duplicate = False
            if frame_id in frames_cache:
                frame_tracks = dict(frames_cache[frame_id])
                for road_tracks in frame_tracks.values():
                    for car_data in road_tracks:
                        is_duplicate = car_util.is_duplicate(car_data, input_data)
                        if is_duplicate:
                            print("发现重复的车辆数据----------")
                            print("对比的车辆数据:{}".format(car_data))
                            print("当前车辆数据:{}".format(input_data))
                            car_tracks.clear()
                            break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break

            # 加入筛选后的轨迹集合
            car_tracks.append(input_data)
        # 如果车辆没有轨迹数据
        if not car_tracks:
            print("原数据车辆[{}]不是有效数据".format(car_id))
            if tracks_dict.get(car_id) != 0:
                tracks_dict.update({car_id: 2})
            continue

        # 如果车辆没有经过注册区和销毁区
        # 不包括视频开始和结束的车辆轨迹
        if not is_in_register or not is_in_destroy:
            #if old_frame_id_arr[0] > 30 and (max_frame == 0 or old_frame_id_arr[-1] < (max_frame - 30)):
                print("原数据车辆[{}]不是完整轨迹，经过注册区：{}，经过销毁区：{}".format(car_id, is_in_register, is_in_destroy))
                if tracks_dict.get(car_id) != 0:
                    tracks_dict.update({car_id: 2})
                continue

        # 如果条件满足，则加入总集合
        if error_num > 3:
            # print("原数据车辆[{}]轨迹异常{}次，大于阈值5次，已丢弃".format(car_id, error_num))
            print("原数据车辆[{}]轨迹异常{}次，大于阈值3次-----".format(car_id, error_num))
            tracks_dict.update({car_id: 2})
            continue

        # 记录完整的车辆轨迹
        tracks_dict.update({car_id: 0})
        """
        添加车辆分组的轨迹缓存集合
        数据结构：
        {
            车辆ID:[车辆轨迹集合]
        }
        """
        cars_cache.update({id: car_tracks})

        """
        添加到每祯的轨迹缓存集合
        数据结构：
        {
            祯ID:{
                 道路ID:[车辆轨迹集合]
            }
        }
        """
        for car_data in car_tracks:
            frame_tracks = {}
            road_tracks = []
            frame_id = car_data.frame_id
            road_id = car_data.road_id
            if frame_id in frames_cache:
                frame_tracks = dict(frames_cache[frame_id])
                if road_id in frame_tracks:
                    road_tracks = list(frame_tracks[road_id])
            road_tracks.append(car_data)
            frame_tracks.update({road_id: road_tracks})
            frames_cache.update({frame_id: frame_tracks})

        # 添加ID映射
        id_dict.update({car_id: id})

    return cars_cache, frames_cache


def handle_step_2(cars_cache, frames_cache):
    """
    第二步数据处理
    前车加速度及extra数据后面计算
    :return:
    """

    print("-----开始执行handle_step_2()")

    # 处理后的轨迹数据缓存（按车辆ID）
    tracks_cache = {}
    for car_id, car_tracks in cars_cache.items():
        # 计算tracks数据
        index = 0
        last_frame_id = 0
        last_x_center = 0
        last_y_center = 0
        last_x_velocity = 0
        last_y_velocity = 0

        # 处理后的数据集合
        tracks_dict = {}
        for car_data in car_tracks:
            id = car_data.id  # 车辆ID
            frame = car_data.frame_id  # 当前帧
            width = car_data.width  # 车辆边界框长度，平均值
            height = car_data.height  # 车辆边界框宽度，平均值
            x_center = car_data.fractional_center_x  # 车辆拟合后的x轴坐标
            y_center = car_data.fractional_center_y  # 车辆拟合后的y轴坐标
            x = x_center - 0.5 * width  # 车辆边界框左上角x轴坐标，X_top_left = X_center – 0.5 * width
            y = y_center - 0.5 * height  # 车辆边界框左上角y轴坐标，拟合后的中心点，Y_top_left = Y_center – 0.5 * height
            x_velocity = car_data.x_velocity  # 车辆x轴速度
            y_velocity = car_data.y_velocity  # 车辆y轴速度
            x_acceleration = car_data.x_a_velocity  # 车辆x轴加速度
            y_acceleration = car_data.y_a_velocity  # 车辆y轴加速度
            lane_id = car_data.road_id  # 车道ID
            up_or_down = car_data.up_or_down  # 上下车道
            # 非旋转AABB
            x_top_left_aabb = x_center - 0.5 * width  # 行驶方向AABB左上角X坐标
            y_top_left_aabb = y_center - 0.5 * height  # 行驶方向AABB左上角Y坐标
            x_top_right_aabb = x_center + 0.5 * width  # 行驶方向AABB右上角X坐标
            y_top_right_aabb = y_center - 0.5 * height  # 行驶方向AABB右上角Y坐标
            x_bottom_right_aabb = x_center + 0.5 * width  # 行驶方向AABB右下角X坐标
            y_bottom_right_aabb = y_center + 0.5 * height  # 行驶方向AABB右下角Y坐标
            x_bottom_left_aabb = x_center - 0.5 * width  # 行驶方向AABB左下角X坐标
            y_bottom_left_aabb = y_center + 0.5 * height  # 行驶方向AABB左下角Y坐标
            # 第一帧的RBB等于AABB
            if index == 0:
                x_top_left_rbb = x_top_left_aabb  # 行驶方向RBB左上角X坐标
                y_top_left_rbb = y_top_left_aabb  # 行驶方向RBB左上角Y坐标
                x_top_right_rbb = x_top_right_aabb  # 行驶方向RBB右上角X坐标
                y_top_right_rbb = y_top_right_aabb  # 行驶方向RBB右上角Y坐标
                x_bottom_right_rbb = x_bottom_right_aabb  # 行驶方向RBB右下角X坐标
                y_bottom_right_rbb = y_bottom_right_aabb  # 行驶方向RBB右下角Y坐标
                x_bottom_left_rbb = x_bottom_left_aabb  # 行驶方向RBB左下角X坐标
                y_bottom_left_rbb = y_bottom_left_aabb  # 行驶方向RBB左下角Y坐标
            else:
                # 旋转RBB
                x1 = np.array([x_center - last_x_center, y_center - last_y_center, 0])  # 车头方向向量
                x1 = x1 / np.linalg.norm(x1)  # 转换单位向量
                z1 = np.array([0, 0, 1])
                y1 = np.cross(x1, z1)  # 车辆横向向量

                rbb_top_left = np.array([x_center, y_center]) - 0.5 * width * x1[:-1] - 0.5 * height * y1[:-1]
                rbb_top_right = np.array([x_center, y_center]) + 0.5 * width * x1[:-1] - 0.5 * height * y1[:-1]
                rbb_bottom_right = np.array([x_center, y_center]) + 0.5 * width * x1[:-1] + 0.5 * height * y1[:-1]
                rbb_bottom_left = np.array([x_center, y_center]) - 0.5 * width * x1[:-1] + 0.5 * height * y1[:-1]
                x_top_left_rbb = rbb_top_left[0]  # 行驶方向RBB左上角X坐标
                y_top_left_rbb = rbb_top_left[1]  # 行驶方向RBB左上角Y坐标
                x_top_right_rbb = rbb_top_right[0]  # 行驶方向RBB右上角X坐标
                y_top_right_rbb = rbb_top_right[1]  # 行驶方向RBB右上角Y坐标
                x_bottom_right_rbb = rbb_bottom_right[0]  # 行驶方向RBB右下角X坐标
                y_bottom_right_rbb = rbb_bottom_right[1]  # 行驶方向RBB右下角Y坐标
                x_bottom_left_rbb = rbb_bottom_left[0]  # 行驶方向RBB左下角X坐标
                y_bottom_left_rbb = rbb_bottom_left[1]  # 行驶方向RBB左下角Y坐标

            # 当前帧所有车辆信息
            frame_tracks = frames_cache[frame]
            # 获取当前车所在车道的轨迹集合
            cur_road_tracks = sorted(dict(frame_tracks).get(car_data.road_id), key=lambda m: m.fractional_center_x)
            cur_index = cur_road_tracks.index(car_data)

            # 八个方向的车辆ID
            preceding_id = 0  # 前车ID
            following_id = 0  # 后车ID
            left_preceding_id = 0  # 左前车ID
            left_alongside_id = 0  # 左中车ID
            left_following_id = 0  # 左后车ID
            right_preceding_id = 0  # 右前车ID
            right_alongside_id = 0  # 右中车ID
            right_following_id = 0  # 右后车ID
            preceding_x_velocity = 0  # 当前车辆前面一辆车的速度
            # 判断是上车道还是下车道，1：上车道  2：下车道
            if car_data.up_or_down == 1:
                # 计算前后路程距离
                front_sight_distance = x
                back_sight_distance = default_config.width / lane_util.pixel_ratio - x

                # 计算前车ID
                if cur_index > 0:
                    preceding_car_data = cur_road_tracks[cur_index - 1]
                    preceding_id = preceding_car_data.id
                # 计算后车ID
                if cur_index < len(cur_road_tracks) - 1:
                    following_id = cur_road_tracks[cur_index + 1].id
                # 计算相邻车道ID
                left_road_id = car_data.road_id + 1
                right_road_id = car_data.road_id - 1
            else:
                front_sight_distance = default_config.width / lane_util.pixel_ratio - x

                back_sight_distance = x
                # 计算前车ID
                if cur_index < len(cur_road_tracks) - 1:
                    preceding_car_data = cur_road_tracks[cur_index + 1]
                    preceding_id = preceding_car_data.id
                # 计算后车ID
                if cur_index > 0:
                    following_id = cur_road_tracks[cur_index - 1].id
                # 计算相邻车道ID
                left_road_id = car_data.road_id - 1
                right_road_id = car_data.road_id + 1

            # 计算左车道相邻车辆ID
            if left_road_id in dict(frame_tracks):
                # 对左车道的车辆轨迹集合进行排序
                left_road_tracks = sorted(dict(frame_tracks).get(left_road_id), key=lambda m: m.fractional_center_x)
                # 获取左车道相邻车辆ID
                left_preceding_id, left_alongside_id, left_following_id = car_util.get_adjacent_car_id(left_road_tracks,
                                                                                                       car_data)
            # 计算右车道相邻车辆ID
            if right_road_id in dict(frame_tracks):
                # 对右车道的车辆轨迹集合进行排序
                right_road_tracks = sorted(dict(frame_tracks).get(right_road_id), key=lambda m: m.fractional_center_x)
                # 获取右车道相邻车辆ID
                right_preceding_id, right_alongside_id, right_following_id = car_util.get_adjacent_car_id(
                    right_road_tracks,
                    car_data)

            # 设置实体属性
            track = Track()
            track.frame = frame
            track.id = id
            track.old_id = car_data.old_id
            track.width = width
            track.height = height
            track.x = x
            track.y = y
            track.x_center = x_center
            track.y_center = y_center
            track.x_velocity = x_velocity
            track.y_velocity = y_velocity
            track.x_acceleration = x_acceleration
            track.y_acceleration = y_acceleration
            track.front_sight_distance = front_sight_distance
            track.back_sight_distance = back_sight_distance
            track.preceding_id = preceding_id
            track.following_id = following_id
            track.left_preceding_id = left_preceding_id
            track.left_alongside_id = left_alongside_id
            track.left_following_id = left_following_id
            track.right_preceding_id = right_preceding_id
            track.right_alongside_id = right_alongside_id
            track.right_following_id = right_following_id
            track.preceding_x_velocity = preceding_x_velocity
            track.lane_id = lane_id
            track.up_or_down = up_or_down
            track.x_top_left_aabb = x_top_left_aabb
            track.y_top_left_aabb = y_top_left_aabb
            track.x_top_right_aabb = x_top_right_aabb
            track.y_top_right_aabb = y_top_right_aabb
            track.x_bottom_right_aabb = x_bottom_right_aabb
            track.y_bottom_right_aabb = y_bottom_right_aabb
            track.x_bottom_left_aabb = x_bottom_left_aabb
            track.y_bottom_left_aabb = y_bottom_left_aabb
            track.x_top_left_rbb = x_top_left_rbb
            track.y_top_left_rbb = y_top_left_rbb
            track.x_top_right_rbb = x_top_right_rbb
            track.y_top_right_rbb = y_top_right_rbb
            track.x_bottom_right_rbb = x_bottom_right_rbb
            track.y_bottom_right_rbb = y_bottom_right_rbb
            track.x_bottom_left_rbb = x_bottom_left_rbb
            track.y_bottom_left_rbb = y_bottom_left_rbb
            track.raw_width = car_data.fractional_bottom_right_x - car_data.fractional_top_left_x
            track.raw_height = car_data.fractional_bottom_right_y - car_data.fractional_top_left_y

            # 加入集合
            tracks_dict.update({frame: track})

            # 设置上一帧的数据
            last_x_center = x_center
            last_y_center = y_center

            # 数据下标
            index = index + 1
        tracks_cache.update({car_id: tracks_dict})
    return tracks_cache


def handle_step_3(tracks_cache):
    """
    第三步数据处理
    前车加速度及extra数据后面计算
    :return:
    """

    print("-----开始执行handle_step_3()")

    for car_id, tracks in tracks_cache.items():
        index = 0  # 下标
        last_lane_id = 0  # 上一帧车道ID
        last_x = 0  # 上一帧车辆轨迹X轴数据
        last_y = 0  # 上一帧车辆轨迹Y轴数据
        for frame_id, track in tracks.items():
            # 获取前车数据
            if track.preceding_id > 0:
                # 前车数据
                preceding_car_data = tracks_cache.get(track.preceding_id).get(frame_id)
                # 前车加速度
                track.preceding_x_velocity = preceding_car_data.x_velocity
                # 计算dhw
                track.dhw = abs(track.x - preceding_car_data.x) - track.width
                # dhw负数处理
                if track.dhw < 0:
                    track.dhw = 0
                # 计算thw
                if abs(track.x_velocity) > 0:
                    track.thw = track.dhw / abs(track.x_velocity)
                # 计算ttc
                if abs(track.x_velocity) - abs(track.preceding_x_velocity) != 0:
                    # 与前车发生碰撞的可能，在后面计算
                    track.ttc = track.dhw / (
                            abs(track.x_velocity) - abs(track.preceding_x_velocity))
                # 计算pet
                track.pet = track.thw

            # 计算变道
            if 0 < last_lane_id != track.lane_id:
                track.lane_keep_intention = 'lane_change'
                # 是否为往左变道
                if (track.up_or_down == 1 and last_lane_id > track.lane_id) or \
                        (track.up_or_down == 2 and last_lane_id < track.lane_id):
                    track.left_lane_change_intention = 'left_lane_change'
            # 计算yaw和steering_angle
            if index > 0:
                l1 = Line(Point(last_x, last_y), Point(track.x, track.y))  # 车辆行驶轨迹线
                l2 = Line(Point(0, 0), Point(1, 0))  # X轴正方向的两个坐标，下车道行驶方向
                l3 = Line(Point(0, 0), Point(-1, 0))  # X轴反方向的两个坐标，上车道行驶方向
                # 计算yaw
                track.yaw = lane_util.line_angle(l1, l3)
                # 计算steering_angle
                if track.up_or_down == 1:
                    track.steering_angle = lane_util.line_angle(l1, l3) / 2
                else:
                    track.steering_angle = lane_util.line_angle(l1, l2) / 2
            # 计算rotation_speed
            track.rotation_speed = track.x_velocity / (2 * np.pi * 0.4) * 360

            # 数据下标
            index = index + 1
    return tracks_cache


def write_csv(tracks_cache, frames_cache, result_path, location_name,video_id, summary_path):
    """
    写入csv文件
    :param tracks_cache: 车辆轨迹数据集合(按车辆ID分组)
    :param frames_cache: 车辆轨迹数据集合(按祯ID分组)
    :param result_path: 输出文件的路径
    :param video_id: 视频ID
    :param summary_path: summary文件路径
    """
    print("开始生成csv文件...")

    # 自动创建目录
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 写入tracks文件
    tracks_file_name = 'tracks.csv'
    tracks_file_path = result_path + tracks_file_name
    tracks_open = open(tracks_file_path, 'w', newline='')
    tracks_writer = csv.writer(tracks_open, dialect='excel')
    tracks_writer.writerow(tracks.get_header())

    # 写入tracks_extra文件
    tracks_extra_file_name = 'tracks_extra.csv'
    tracks_extra_file_path = result_path + tracks_extra_file_name
    tracks_extra_open = open(tracks_extra_file_path, 'w', newline='')
    tracks_extra_writer = csv.writer(tracks_extra_open, dialect='excel')
    tracks_extra_writer.writerow(tracks_extra.get_header())

    # tracks数据集合
    all_tracks = {}

    # 新的frame_id序列
    new_frame_id = 1
    frame_id_dict = {}
    old_frame_arr = sorted(frames_cache.keys())
    for frame_id in old_frame_arr:
        frame_id_dict.update({frame_id: new_frame_id})
        new_frame_id = new_frame_id + 1

    # 遍历数据
    for car_id, car_tracks in tracks_cache.items():
        car_tracks_list = []
        for frame_id, track in car_tracks.items():
            if default_config.debug:
                frame = track.frame
            else:
                frame = frame_id_dict.get(frame_id)

            # 写入tracks文件
            tracks_data = [frame,
                           track.id,
                           round(track.x, 2),
                           round(track.y, 2),
                           round(track.width, 2),
                           round(track.height, 2),
                           round(track.x_velocity, 4),
                           round(track.y_velocity, 4),
                           round(track.x_acceleration, 4),
                           round(track.y_acceleration, 4),
                           round(track.front_sight_distance, 2),
                           round(track.back_sight_distance, 2),
                           round(track.dhw, 2),
                           round(track.thw, 2),
                           round(track.ttc, 2),
                           round(track.preceding_x_velocity, 2),
                           track.preceding_id,
                           track.following_id,
                           track.left_preceding_id,
                           track.left_alongside_id,
                           track.left_following_id,
                           track.right_preceding_id,
                           track.right_alongside_id,
                           track.right_following_id,
                           track.lane_id,
                           round(track.x_top_left_aabb, 2),
                           round(track.y_top_left_aabb, 2),
                           round(track.x_top_right_aabb, 2),
                           round(track.y_top_right_aabb, 2),
                           round(track.x_bottom_right_aabb, 2),
                           round(track.y_bottom_right_aabb, 2),
                           round(track.x_bottom_left_aabb, 2),
                           round(track.y_bottom_left_aabb, 2),
                           round(track.x_top_left_rbb, 2),
                           round(track.y_top_left_rbb, 2),
                           round(track.x_top_right_rbb, 2),
                           round(track.y_top_right_rbb, 2),
                           round(track.x_bottom_right_rbb, 2),
                           round(track.y_bottom_right_rbb, 2),
                           round(track.x_bottom_left_rbb, 2),
                           round(track.y_bottom_left_rbb, 2),
                           round(track.raw_width, 2),
                           round(track.raw_height, 2),
                           track.old_id,
                           round(track.x_center, 2),
                           round(track.y_center, 2)
                           ]
            track.frame = frame
            car_tracks_list.append(track)
            tracks_writer.writerow(tracks_data)

            # 写入tracks_extra文件
            tracks_extra_data = [frame,
                                 track.id,
                                 round(track.pet, 2),
                                 track.lane_keep_intention,
                                 track.left_lane_change_intention,
                                 track.take_over_intention,
                                 track.yaw,
                                 track.roll,
                                 track.pitch,
                                 round(track.rotation_speed, 2),
                                 track.angular_velocity,
                                 track.steering_angle]
            tracks_extra_writer.writerow(tracks_extra_data)

        all_tracks.update({car_id: car_tracks_list})

    print("tracks和tracks_extra文件生成完毕！")

    # 写入tracksMeta文件和recordingMeta文件
    tracks_meta_file_name = 'tracksMeta.csv'
    tracks_meta_file_path = result_path + tracks_meta_file_name

    recording_meta_file_name = 'recordingMeta.csv'
    recording_meta_file_path = result_path + recording_meta_file_name

    # summary_file_name = 'data_summary_cyf.csv'
    # summary_file_path = result_path + summary_file_name

    tracks_mate_util.create_tracks_and_recording_mata(all_tracks, tracks_meta_file_path, recording_meta_file_path,
                                                      summary_path, location_name, video_id)
    print("tracksMeta文件和recordingMeta文件生成完毕！")


def read_from_local(file_name, chunk_size=50000):
    reader = pandas.read_csv(file_name, header=0, iterator=True, encoding="utf-8")
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped!")
    # 将块拼接为pandas dataFrame格式
    df_ac = pandas.concat(chunks, ignore_index=True)

    return df_ac


def write_statistics_csv(result_path):
    header = ["id", "status"]

    # 写入statistics文件
    statistics_file_name = 'statistics.csv'
    statistics_file_path = result_path + statistics_file_name
    statistics_open = codecs.open(statistics_file_path, 'w', 'gbk')
    statistics_writer = csv.writer(statistics_open, dialect='excel')
    statistics_writer.writerow(header)

    total = len(tracks_dict.keys())
    processed = 0
    filtered = 0
    tag_true = 0
    other = 0
    for car_id, status in tracks_dict.items():
        data = [car_id,
                status
                ]
        statistics_writer.writerow(data)

        if status == 0:
            processed = processed + 1
        elif status == 1:
            tag_true = tag_true + 1
        elif status == 2:
            filtered = filtered + 1
        else:
            other = other + 1
    print("statistics文件生成完毕！")

    # 写入statistics_ext文件
    header = ["total", "tag_true", "processed", "filtered", "other", "processed_rate", "precision"]
    statistics_ext_file_name = 'statistics_ext.csv'
    statistics_ext_file_path = result_path + statistics_ext_file_name
    statistics_ext_open = open(statistics_ext_file_path, 'w', newline='')
    statistics_ext_writer = csv.writer(statistics_ext_open, dialect='excel')
    statistics_ext_writer.writerow(header)
    # 计算处理比例
    processed_rate = processed / total * 100
    # 计算准确率
    precision = processed / (processed + tag_true) * 100
    data = [total, tag_true, processed, filtered, other, processed_rate, precision]
    statistics_ext_writer.writerow(data)
    print("statistics_ext文件生成完毕！")
    print("total:", total)
    print("tag_true:", tag_true)
    print("processed:", processed)
    print("filtered:", filtered)
    print("other:", other)
    print("processed_rate:", processed_rate)
    print("precision:", precision)


def kalman_filter(z):
    '''
    :param z: 待滤波的数组
    :return: 滤波之后的数组
    '''
    # intial parameters
    n_iter = len(z)
    sz = (n_iter,)  # size of array
    x = -0.37727  # truth value (typo in example at top of p. 13 calls this z)
    Q = 1e-5  # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = 0.1 ** 2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = z[0]
    P[0] = 1.0

    for k in range(1, len(z)):
        # time update
        xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat
