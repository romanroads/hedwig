# !/usr/bin/python
# coding=utf-8

debug = False  # debug模式
fps = 30  # 每秒帧数
width = 3840  # 背景宽度，单位像素
height = 2160  # 背景高度，单位像素
register_length = 440  # 注册区宽度，单位像素
destroy_length = 440  # 销毁区宽度，单位像素
car_bb_fixed = 0.32  # 车辆边框修正值，单位米
xa_range = [2, -2]  # x加速度范围，单位米/s
ya_range = [1, -1]  # y加速度范围，单位米/s
car_height_range = [1.55, 1.85]  # 轿车宽度范围，单位米
truck_height_range = [1.8, 2.5]  # 货车宽度范围，单位米

