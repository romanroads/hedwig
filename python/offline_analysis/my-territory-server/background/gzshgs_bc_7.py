# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(补采)-7）
"""
default = {
    1: [923, 952],
    2: [948, 978],
    3: [980, 1008],
    4: [1013, 1040],
    5: [1046, 1072],
    6: [1069, 1092],
    7: [1099, 1122],
    8: [1131, 1154],
    9: [1163, 1187],
    10: [1189, 1212]
}
pixel_ratio = 8.27  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.48174396418769, 2.38477642311667]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.16598100150175, 2.65312444283333]  # 经过统计后的货车宽度范围
