# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州广台高速-4）
"""
default = {
    1: [930, 927],
    2: [965, 965],
    3: [999, 996],
    4: [1032, 1028],
    5: [1065, 1060],
    6: [1105, 1098],
    7: [1137, 1130],
    8: [1170, 1164],
    9: [1202, 1195],
    10: [1238, 1235]
}
pixel_ratio = 8.53  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.49705129104945, 2.36138946469293]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.22990898794854, 2.75190822215477]  # 经过统计后的货车宽度范围