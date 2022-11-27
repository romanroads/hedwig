# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(补采)-16）
"""
default = {
    1: [894, 968],
    2: [920, 994],
    3: [952, 1026],
    4: [985, 1058],
    5: [1017, 1090],
    6: [1041, 1111],
    7: [1073, 1142],
    8: [1104, 1174],
    9: [1136, 1208],
    10: [1162, 1236]
}
pixel_ratio = 8.53  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.57100838734837, 2.39983289767843]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.21555572006404, 2.6235336748648]  # 经过统计后的货车宽度范围