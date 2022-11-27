# !/usr/bin/python
# coding=utf-8

import importlib

import numpy as np
import pandas
from matplotlib import pyplot as plt

from config import default_config
from util import data_util, lane_util


def is_last(reader):
    try:
        next(reader)
        return False
    except StopIteration:
        return True


def generate(path, gc_background, result_type, result_path, location_name, video_id, summary_path):
    """
    文件生成
    :param path:输入文件路径
    :param gc_background: 道路背景设置
    :param result_type: 生成结果类型 all：全部 img：图片 csv：文件
    :param result_path: 输出文件路径
    :param location_name: 文件夹名称
    :param video_id: 视频ID
    :param summary_path: summary文件路径
    :return
    """

    # 道路背景设置
    lane_util.set_background(gc_background)
    for lane_id, lane in lane_util.default_lanes.items():
        print("lane_id:{}".format(lane_id))
        print(lane)

    # 重置全局变量
    data_util.tracks_dict = {}
    data_util.max_frame = 0
    data_util.id_dict = {}

    # 分批处理
    chunk_size = 100000
    prev_filter_data = None
    all_cars_cache, all_frames_cache, all_result = {}, {}, {}
    num = 0
    # 第一次读取，计算总片数
    with pandas.read_csv(path, chunksize=chunk_size) as reader:
        for chunk in reader:
            num += 1
        print("csv总分片数：{}".format(num))
        reader.close()

    # 第二次读取，处理文件
    index = 0
    with pandas.read_csv(path, chunksize=chunk_size) as reader:
        print("-----开始读取csv文件：{}".format(path))
        for chunk in reader:
            index += 1
            print("-----第{}次分批处理-----".format(index))
            # 判断是否为最后一批
            isLast = False
            if index == num:
                isLast = True
            print("是否为最后一批数据：{}".format(isLast))
            # csv数据筛选
            filter_data = data_util.pre_filter(chunk, isLast)
            if prev_filter_data is not None:
                filter_data_combined = prev_filter_data + filter_data
            else:
                filter_data_combined = filter_data
            prev_filter_data = filter_data

            # 数据处理
            cars_cache, frames_cache = None, None
            if filter_data_combined:
                cars_cache, frames_cache = data_util.handle_step_1(filter_data_combined)

            if cars_cache and frames_cache:
                # 合并到总的结果记录
                all_cars_cache.update(cars_cache)
                all_frames_cache.update(frames_cache)
                # 后续数据处理
                tracks_cache = data_util.handle_step_2(cars_cache, frames_cache)
                result = data_util.handle_step_3(tracks_cache)
                # 合并到总的结果记录
                all_result.update(result)

    print("筛选后的总车数：{}".format(len(all_cars_cache)))
    # 生成csv结果文件
    if result_type == 'all' or result_type == 'csv':
        if all_result and all_frames_cache and all_cars_cache:
            data_util.write_csv(all_result, all_frames_cache, result_path, location_name, video_id, summary_path)

    # 生成轨迹图
    if result_type == 'all' or result_type == 'img':
        if all_frames_cache and all_cars_cache:
            # generate_tracks_img1(filter_data_combined, all_cars_cache, result_path)
            generate_tracks_img(all_cars_cache, result_path)

            # 生成x速度图
            generate_xv_img(all_cars_cache, result_path)

            # 生成y速度图
            generate_yv_img(all_cars_cache, result_path)

            # 生成x加速度图
            generate_xva_img(all_cars_cache, result_path)

            # 生成y加速度图
            generate_yva_img(all_cars_cache, result_path)

    #生成统计文件
    if result_type == 'all' or result_type == 'statistics':
        if all_frames_cache and all_cars_cache:
            data_util.write_statistics_csv(result_path)


def generate_tracks_img(cars_cache, result_path):
    """
    绘制轨迹图
    :param cars_cache: 车辆轨迹集合
    :param result_path: 输出文件路径
    """

    print("开始生成png文件...")

    # 设置图片文件名
    img_name = "tracks_chart.png"
    img_path = result_path + img_name

    # 道路设置
    for lane_id, lane in lane_util.default_lanes.items():
        lane_x = np.linspace(0, default_config.width / lane_util.pixel_ratio, 100)
        lane_y = [lane.top_left_y] * len(lane_x)
        if lane_id in [1, 2, 6, 9]:
            plt.plot(lane_x, lane_y, color='white', linewidth=0.5, linestyle='-')
        else:
            plt.plot(lane_x, lane_y, color='white', linewidth=0.5, linestyle='--')

        if lane_id in [4, 9]:
            lane_y = [lane.bottom_left_y] * len(lane_x)
            plt.plot(lane_x, lane_y, color='white', linewidth=0.5, linestyle='-')

        if lane_id in [1, 9]:
            plt.fill_between(lane_x, lane.top_left_y, lane.bottom_left_y, facecolor='silver', alpha=1)
        else:
            plt.fill_between(lane_x, lane.top_left_y, lane.bottom_left_y, facecolor='darkgray', alpha=1)

    for car_id, car_group_data in cars_cache.items():
        x = [car_data.fractional_center_x for car_data in car_group_data]
        y = [car_data.fractional_center_y for car_data in car_group_data]
        plt.plot(x, y, linewidth=0.5)
        #plt.text(x[0], y[0], car_id, ha="right", va="bottom")  # 格式化字符串，保留0位小数

    plt.xlabel("x-Coordinate")
    plt.ylabel("y-Coordinate")
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().set_facecolor('yellowgreen')
    plt.xlim(0 + lane_util.exclude_left_x, default_config.width / lane_util.pixel_ratio)
    first_y = lane_util.default_lanes.get(1).top_left_y - 10
    last_y = lane_util.default_lanes.get(9).bottom_left_y + 10
    plt.ylim(last_y, first_y)
    #plt.ylim(default_config.height, 0)
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print("png文件生成完毕！")


def generate_tracks_img1(filter_data_combined, cars_cache, result_path):
    """
    绘制轨迹图
    :param filter_data_combined: 原始车辆轨迹集合
    :param cars_cache: 车辆轨迹集合
    :param result_path: 输出文件路径
    """

    print("开始生成png文件...")

    # 设置图片文件名
    img_name = "tracks_chart.png"
    img_path = result_path + img_name

    """
    for lane_id, lane in lane_util.default_lanes.items():
        print("lane_id:{}".format(lane_id))
        print(lane)
        lane_x = np.linspace(0, default_config.forecast, 100)
        lane_y = [lane.top_left_y] * len(lane_x)
        if lane_id in [1, 2, 6, 9]:
            plt.plot(lane_x, lane_y, color='white', linewidth=0.5, linestyle='-')
        else:
            plt.plot(lane_x, lane_y, color='white', linewidth=0.5, linestyle='--')

        if lane_id in [4, 9]:
            lane_y = [lane.bottom_left_y] * len(lane_x)
            plt.plot(lane_x, lane_y, color='white', linewidth=0.5, linestyle='-')

        if lane_id in [1, 9]:
            plt.fill_between(lane_x, lane.top_left_y, lane.bottom_left_y, facecolor='silver', alpha=1)
        else:
            plt.fill_between(lane_x, lane.top_left_y, lane.bottom_left_y, facecolor='darkgray', alpha=1)
    """

    # 滤波后的轨迹
    for car_id, car_group_data in cars_cache.items():
        x = [car_data.fractional_center_x for car_data in car_group_data]
        y = [car_data.fractional_center_y for car_data in car_group_data]
        plt.plot(x, y, linewidth=0.5, label="Filter Tracks")
        # plt.text(x[0], y[0], car_id, ha="center", va="bottom")  # 格式化字符串，保留0位小数

    # 原始数据
    x = [row['fractional_center_x'] for row in filter_data_combined]
    y = [row['fractional_center_y'] for row in filter_data_combined]
    plt.plot(x, y, linewidth=0.5, label="Original Tracks")

    plt.xlabel("x-Coordinate")
    plt.ylabel("y-Coordinate")
    plt.gca().xaxis.set_ticks_position('top')
    #plt.gca().set_facecolor('yellowgreen')
    plt.xlim(0 + lane_util.exclude_left_x, default_config.width / lane_util.pixel_ratio)
    first_y = lane_util.default_lanes.get(1).top_left_y - 10
    last_y = lane_util.default_lanes.get(9).bottom_left_y + 10
    #plt.ylim(last_y, first_y)
    #plt.ylim(default_config.height, 0)
    plt.legend()
    plt.savefig(img_path)
    print("轨迹图png文件生成完毕！")


def generate_xv_img(cars_cache, result_path):
    """
    绘制x速度图
    :param cars_cache: 车辆轨迹集合
    """

    print("开始生成x速度png文件...")

    # 设置图片文件名
    img_name = "x_v_chart.png"
    img_path = result_path + img_name

    # 滤波后的轨迹
    for car_id, car_group_data in cars_cache.items():
        x = [car_data.fractional_center_x for car_data in car_group_data]
        y = [car_data.x_velocity for car_data in car_group_data]
        plt.plot(x, y, linewidth=0.5)

    plt.xlabel("Center X")
    plt.ylabel("X Velocity")
    plt.gca().xaxis.set_ticks_position('top')
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print("x速度图png文件生成完毕！")


def generate_xva_img(cars_cache, result_path):
    """
    绘制x加速度图
    :param cars_cache: 车辆轨迹集合
    """

    print("开始生成x加速度png文件...")

    # 设置图片文件名
    img_name = "x_a_v_chart.png"
    img_path = result_path + img_name

    # 滤波后的轨迹
    for car_id, car_group_data in cars_cache.items():
        x = [car_data.fractional_center_x for car_data in car_group_data]
        y = [car_data.x_a_velocity for car_data in car_group_data]
        plt.plot(x, y, linewidth=0.5)

    plt.xlabel("Center X")
    plt.ylabel("X Acceleration")
    plt.gca().xaxis.set_ticks_position('top')
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print("x加速度图png文件生成完毕！")


def generate_yv_img(cars_cache, result_path):
    """
    绘制y速度图
    :param cars_cache: 车辆轨迹集合
    """

    print("开始生成y速度png文件...")
    # 设置图片文件名
    img_name = "y_v_chart.png"
    img_path = result_path + img_name

    # 滤波后的轨迹
    for car_id, car_group_data in cars_cache.items():
        x = [car_data.fractional_center_x for car_data in car_group_data]
        y = [car_data.y_velocity for car_data in car_group_data]
        plt.plot(x, y, linewidth=0.5)


    plt.xlabel("Center X")
    plt.ylabel("Y Velocity")
    plt.gca().xaxis.set_ticks_position('top')
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print("y速度图png文件生成完毕！")


def generate_yva_img(cars_cache, result_path):
    """
    绘制y加速度图
    :param cars_cache: 车辆轨迹集合
    """

    print("开始生成y加速度png文件...")

    # 设置图片文件名
    img_name = "y_a_v_chart.png"
    img_path = result_path + img_name

    # 滤波后的轨迹
    for car_id, car_group_data in cars_cache.items():
        x = [car_data.fractional_center_x for car_data in car_group_data]
        y = [car_data.y_a_velocity for car_data in car_group_data]
        plt.plot(x, y, linewidth=0.5)

    plt.xlabel("Center X")
    plt.ylabel("Y Acceleration")
    plt.gca().xaxis.set_ticks_position('top')
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print("y加速度图png文件生成完毕！")
