# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州环城高速-1）
"""
default = {
    1: [885, 896],
    2: [915, 926],
    3: [950, 960],
    4: [985, 992],
    5: [1020, 1025],
    6: [1050, 1053],
    7: [1084, 1084],
    8: [1118, 1117],
    9: [1153, 1150],
    10: [1184, 1180]
}
pixel_ratio = 8.8  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.57640767125067, 2.54044381247852]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.32084055363185, 2.7983653795394]  # 经过统计后的货车宽度范围
