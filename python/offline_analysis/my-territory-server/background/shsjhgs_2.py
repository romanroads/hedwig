# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（上海申嘉湖高速-2）
"""
default = {
    1: [854, 906],
    2: [877, 933],
    3: [910, 965],
    4: [943, 997],
    5: [975, 1030],
    6: [1052, 1107],
    7: [1083, 1139],
    8: [1114, 1171],
    9: [1145, 1203],
    10: [1173, 1232]
}
pixel_ratio = 8.53  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.75647627698329, 2.53972187479562]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.34018973644189, 3.21825788404422]  # 经过统计后的货车宽度范围