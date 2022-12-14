# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(补采)-3）
"""
default = {
    1: [921, 962],
    2: [945, 988],
    3: [977, 1019],
    4: [1009, 1050],
    5: [1041, 1080],
    6: [1064, 1100],
    7: [1094, 1130],
    8: [1126, 1161],
    9: [1157, 1193],
    10: [1182, 1220]
}
pixel_ratio = 8.27  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.44714744298546, 2.32728272935013]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.14873919208287, 2.63149556122566]  # 经过统计后的货车宽度范围
