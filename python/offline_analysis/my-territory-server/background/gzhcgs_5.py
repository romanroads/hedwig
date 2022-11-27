# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州环城高速-5）
"""
default = {
    1: [880, 974],
    2: [910, 1004],
    3: [943, 1037],
    4: [977, 1072],
    5: [1011, 1107],
    6: [1039, 1138],
    7: [1074, 1173],
    8: [1107, 1207],
    9: [1142, 1240],
    10: [1170, 1270]
}
pixel_ratio = 8.8  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.77832670840018, 2.63328478154758]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.46034081161828, 2.89150294741062]  # 经过统计后的货车宽度范围
