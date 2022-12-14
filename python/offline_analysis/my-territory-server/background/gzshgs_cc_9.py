# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(初采)-9）
"""
default = {
    1: [916, 935],
    2: [941, 961],
    3: [971, 991],
    4: [1002, 1021],
    5: [1034, 1051],
    6: [1057, 1070],
    7: [1085, 1098],
    8: [1117, 1128],
    9: [1147, 1157],
    10: [1173, 1183]
}
pixel_ratio = 8  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.36656284063391, 2.2573518018859]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.09658408294579, 2.57968233931811]  # 经过统计后的货车宽度范围
