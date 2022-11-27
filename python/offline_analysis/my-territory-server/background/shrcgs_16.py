# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（上海绕城高速-16）
"""
default = {
    1: [923, 935],
    2: [953, 964],
    3: [985, 996],
    4: [1017, 1029],
    5: [1049, 1061],
    6: [1091, 1104],
    7: [1123, 1136],
    8: [1155, 1169],
    9: [1187, 1202],
    10: [1213, 1230]
}
pixel_ratio = 8.53  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.54146648876363, 2.52462925924779]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.235767288862, 3.18466170196393]  # 经过统计后的货车宽度范围
