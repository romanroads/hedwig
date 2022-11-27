# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州环城高速-2.1）
"""
default = {
    1: [909, 940],
    2: [939, 970],
    3: [972, 1004],
    4: [1006, 1039],
    5: [1040, 1075],
    6: [1069, 1102],
    7: [1100, 1137],
    8: [1135, 1171],
    9: [1169, 1206],
    10: [1198, 1238]
}
pixel_ratio = 8.8  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.58319660978724, 2.51825909694604]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.29820294119266, 2.76662189339732]  # 经过统计后的货车宽度范围