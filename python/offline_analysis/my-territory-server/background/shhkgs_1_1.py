# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（上海沪昆高速-1.1）
"""
default = {
    1: [925, 940],
    2: [950, 964],
    3: [987, 998],
    4: [1022, 1032],
    5: [1057, 1065],
    6: [1095, 1100],
    7: [1129, 1133],
    8: [1164, 1167],
    9: [1201, 1201],
    10: [1225, 1224]
}
pixel_ratio = 9.33  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.43429293994767, 2.05556366442074]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.40773261844219, 2.72019556630557]  # 经过统计后的货车宽度范围
