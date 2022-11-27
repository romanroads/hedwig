# !/usr/bin/python
# coding=utf-8

import csv

import numpy as np

from config import default_config
from output import tracks
from output.tracks_meta import TracksMeta
from output.recording_meta import RecordingMeta
from util import lane_util
import pandas


def create_tracks_and_recording_mata(all_tracks, tracks_meta_file_path, recording_meta_file_path, summary_file_path, location_name,video_id):
    """
    生成tracksMeta.csv和recordingMeta.csv
    :param all_tracks: 所有车辆轨迹
    :param tracks_meta_file_path: tracks_mata.csv文件完整路径
    :param recording_meta_file_path: recording_meta.csv文件完整路径
    :param summary_file_path: summary.csv文件完整路径
    :param video_id: 视频ID
    :return:
    """

    # 生成统计数据
    count_data = create_count_data_by_tracks_data(all_tracks)
    # 解析并查找summary.csv文件
    summary_data = read_summary_csv(summary_file_path, location_name, video_id)
    # 生成tracksMeta.csv
    create_tracks_meta_csv(count_data, tracks_meta_file_path)
    # 生成recordingMeta.csv文件
    create_recording_meta_csv(count_data, recording_meta_file_path, summary_data)


def create_count_data_by_tracks_data(tracks_data):
    """
    根据封装好的 tracks 集合数据生成统计数据
    :param: tracks_data tracks封装的数据
    :return: 统计后的集合数据
    """
    if not tracks_data:
        raise Exception("tracks的数据不能为空")

    # tracks_data 数据模型 {1:[{k1:v1,k2:v2...},{k1:v1,k2:v2...}],2:[{k1:v1,k2:v2...},{k1:v1,k2:v2...}]}
    count_data = []
    for car_id in tracks_data:
        car_info = tracks_data[car_id]
        if car_info:
            compute_tracks_data(car_info, count_data, car_id)
        else:
            pass

    return count_data


def compute_tracks_data(car_infos, count_data, car_id):
    """
    根据每一辆车的全部帧数的信息来计算出统计数据并存入集合
    :param car_infos:  每一个Id的车的全部帧的信息
    :param count_data: 统计的集合数据
    :param car_id: 车辆Id
    """
    # 第一帧和最后一帧的数据
    first_car_info = car_infos[0]
    last_car_info = car_infos[len(car_infos) - 1]
    min_x_velocity = first_car_info.x_velocity  # 最小速度 默认第一帧速度
    max_x_velocity = first_car_info.x_velocity  # 最大速度 默认第一帧速度
    min_dhw = -1
    min_thw = -1
    min_ttc = -1
    total_x_velocity = 0  # 累计车速
    lane_arr = []
    before_num_lane = first_car_info.lane_id  # 上一帧的车道 默认第一帧的车道
    num_lane_changes = 0

    # 遍历单个车辆所有帧的数据 计算出一些统计数据
    for index, car_info in enumerate(car_infos):
        # 查找最大车速
        if abs(float(car_info.x_velocity)) > abs(float(max_x_velocity)):
            max_x_velocity = car_info.x_velocity

        # 查找最小车速
        if abs(float(car_info.x_velocity)) < abs(float(min_x_velocity)):
            min_x_velocity = car_info.x_velocity

        # 累计车速9
        total_x_velocity += float(car_info.x_velocity)

        # 获取车道ID集合
        lane_arr.append(car_info.lane_id)

        # 换道次数
        if car_info.lane_id != before_num_lane:
            num_lane_changes += 1
            before_num_lane = car_info.lane_id

        # 最小dhw
        dhw = float(car_info.dhw)
        if dhw != 0:
            if min_dhw == -1:
                min_dhw = dhw
            else:
                if dhw < min_dhw:
                    min_dhw = dhw

        # 最小thw
        thw = float(car_info.thw)
        if thw != 0:
            if min_thw == -1:
                min_thw = thw
            else:
                if thw < min_thw:
                    min_thw = thw

        # 最小ttc
        ttc = float(car_info.ttc)
        if ttc > 0:
            if min_ttc == -1:
                min_ttc = ttc
            else:
                if ttc < float(min_ttc):
                    min_ttc = ttc

    # numFrames
    numFrames = car_infos[-1].frame - car_infos[0].frame + 1

    # 生成 TracksMeta 对象并保存到 tracks_mates 集合中
    count_data.append({"id": car_id,  # id 车辆 ID
                       "width": first_car_info.width,  # width 车辆边界框长度
                       "height": first_car_info.height,  # height 车辆边界框宽度
                       "initialFrame": first_car_info.frame,  # initialFrame 车辆轨迹开始的初始帧
                       "finalFrame": last_car_info.frame,  # finalFrame 车辆轨迹结束帧
                       "numFrames": numFrames,  # numFrames 车辆轨迹总计帧数
                       "classes": "Car" if float(first_car_info.width) <= 8 else "Truck",
                       # class 车辆类型（car、truck）
                       "drivingDirection": 1 if int(float(first_car_info.lane_id)) <= 4 else 2,
                       # drivingDirection 车辆的行驶方向。左方向（上车道）为1，右方向（下车道）为2
                       "traveledDistance": abs(float(first_car_info.x_top_left_aabb) - float(
                           last_car_info.x_top_left_aabb)),
                       # traveledDistance 高速公路路段上车辆行驶过的距离
                       "minXVelocity": abs(float(min_x_velocity)),  # minXVelocity 行驶方向的最小速度
                       "maxXVelocity": abs(float(max_x_velocity)),  # maxXVelocity 行驶方向的最大速度
                       "meanXVelocity": abs(total_x_velocity / len(car_infos)),
                       # meanXVelocity 行驶方向的平均速度
                       "minDHW": min_dhw,  # minDHW 最小行车间隔（DHW）。如果前面没有车辆，则此值设置为-1
                       "minTHW": min_thw,  # minTHW 最小行车间隔（THW）。如果前面没有车辆，则此值设置为-1
                       "minTTC": min_ttc,  # minTTC 最小碰撞时间（TTC）。如果不存在前车或有效的TTC，则此值设置为-1
                       "numLaneChanges": num_lane_changes  # numLaneChanges 换道次数
                       })


def create_tracks_meta_csv(count_data, tracks_meta_path):
    """
    创建tracksMeta.csv文件
    :param count_data: 统计的数据
    :param tracks_meta_path: 文件保存的路径
    :return:
    """
    if count_data:
        tracks_meta_file = open(tracks_meta_path, 'w', newline="", encoding="utf-8")
        print("生成的tracksMeta.csv文件地址: %s" % tracks_meta_path)
        csv_write = csv.writer(tracks_meta_file)
        # 写入csv的表头
        csv_write.writerow(TracksMeta.create_csv_title())
        # 插入每一行的数据
        for i in count_data:
            csv_write.writerow(TracksMeta(i).case_to_list())
        tracks_meta_file.close()


def create_recording_meta_csv(count_data, recording_meta_path, summary_data):
    """
    创建recordingMeta.csv文件
    :param summary_data: 一些字段的数据来源
    :param count_data: 统计的数据
    :param recording_meta_path: 文件保存的路径
    :return:
    """
    if count_data:
        # 计算recording_meta的数据
        total_driven_distance = 0  # 所有车辆行驶距离总和
        total_driven_time = 0  # 所有车辆行驶时间的总和
        num_vehicles = 0  # 视频中车辆总数
        num_cars = 0  # 视频中小车总数
        num_trucks = 0  # 视频中大车总数
        max_frame = 0  # 最大帧数
        for i in count_data:
            num_vehicles += 1

            if id(i["classes"]) == id("Car"):
                num_cars += 1
            else:
                num_trucks += 1

            total_driven_distance += i["traveledDistance"]

            total_driven_time += i["numFrames"]

            # 获取最大的frame_id
            if i["finalFrame"] > max_frame:
                max_frame = i["finalFrame"]

        # 计算上车道和下车道线
        upper_lane_markings = ""
        lower_lane_markings = ""
        for lane_id, lane in lane_util.default_lanes.items():
            if lane.up_or_down == 1:
                # 保留应急车道外侧车道线
                if lane_id == 1:
                    upper_lane_markings += str(cut(lane.top_left_y, 2)) + ";"
                upper_lane_markings += str(cut(lane.bottom_left_y, 2)) + ";"
            else:
                lower_lane_markings += str(cut(lane.top_left_y, 2)) + ";"
                # 保留应急车道外侧车道线
                if lane_id == 9:
                    lower_lane_markings += str(cut(lane.bottom_left_y, 2)) + ";"
        tracks_meta_file = open(recording_meta_path, 'w', newline="", encoding="utf-8")
        print("生成的recordingMeta.csv文件地址: %s" % recording_meta_path)
        csv_write = csv.writer(tracks_meta_file)
        # 写入csv的表头
        csv_write.writerow(RecordingMeta.create_csv_title())
        csv_write.writerow(RecordingMeta({"id": summary_data["id"],
                                          "frameRate": default_config.fps,
                                          "locationId": summary_data["locationId"],
                                          "speedLimit": int(summary_data["speedLimit"]) * 1000 / 3600,
                                          "month": summary_data["month"],
                                          "weekDay": summary_data["weekDay"],
                                          "startTime": summary_data["startTime"],
                                          "duration": round(max_frame / float(default_config.fps), 2),
                                          "totalDrivenDistance": round(
                                              float(total_driven_distance),
                                              2),
                                          "totalDrivenTime": format(
                                              float(total_driven_time) / float(default_config.fps), '.2f'),
                                          "numVehicles": num_vehicles,
                                          "numCars": num_cars,
                                          "numTrucks": num_trucks,
                                          "upperLaneMarkings": upper_lane_markings[0: -1],
                                          "lowerLaneMarkings": lower_lane_markings[0: -1],
                                          "frameWidth": round(float(default_config.width) / lane_util.pixel_ratio,
                                                              2),
                                          "frameHeight": round(
                                              float(default_config.height) / lane_util.pixel_ratio,
                                              2)}).case_to_list())


def read_summary_csv(summary_csv_path, location_name, video_id):
    """
    读取总结文件获取对应video_id的数据
    :param video_id: 目录名称
    :param video_id: 视频Id
    :param summary_csv_path:
    :return: 返回根据videoId封装好的数据
    """

    # 读取总结csv文件
    with open(summary_csv_path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            locationId = row[0]
            locationName = row[1]
            fileName = row[2]
            speedLimit = row[3]
            month = row[4]
            weekDay = row[5]
            startTime = row[6]
            gpsX = row[7]
            gpsY = row[8]
            id = row[9]

            if locationName == location_name and fileName == video_id:
                return {
                    "locationId": locationId,
                    "locationName": locationName,
                    "fileName": fileName,
                    "speedLimit": speedLimit,
                    "month": month,
                    "weekDay": weekDay,
                    "startTime": startTime,
                    "gpsX": gpsX,
                    "gpsY": gpsY,
                    "id": id
                }

def cut(num, c):
    """
    保留c位小数 不四舍五入
    :param num: 数值
    :param c: 保留小数位
    :return:
    """
    str_num = str(num)
    return float(str_num[:str_num.index('.') + 1 + c])
