# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（上海沪昆高速-4）
"""
default = {
    1: [922, 943],
    2: [948, 969],
    3: [983, 1002],
    4: [1018, 1035],
    5: [1052, 1066],
    6: [1090, 1101],
    7: [1126, 1133],
    8: [1161, 1166],
    9: [1197, 1200],
    10: [1222, 1224]
}
pixel_ratio = 9.33  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.39002180810192, 2.3167722830969]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.10501234727557, 2.75510783661032]  # 经过统计后的货车宽度范围