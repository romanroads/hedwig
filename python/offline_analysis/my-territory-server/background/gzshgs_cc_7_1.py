# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(初采)-7.1）
"""
default = {
    1: [955, 958],
    2: [978, 981],
    3: [1009, 1011],
    4: [1039, 1040],
    5: [1068, 1070],
    6: [1090, 1088],
    7: [1117, 1116],
    8: [1147, 1146],
    9: [1177, 1175],
    10: [1200, 1201]
}
pixel_ratio = 8  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.33117140862351, 2.29718631413499]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.01732670917096, 2.59933095626264]  # 经过统计后的货车宽度范围
