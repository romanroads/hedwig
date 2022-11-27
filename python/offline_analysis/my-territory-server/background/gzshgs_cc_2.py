# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(初采)-2）
"""
default = {
    1: [904, 958],
    2: [929, 984],
    3: [961, 1015],
    4: [993, 1046],
    5: [1025, 1075],
    6: [1049, 1095],
    7: [1079, 1124],
    8: [1112, 1155],
    9: [1142, 1186],
    10: [1168, 1212]
}
pixel_ratio = 8.27  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.49245828369436, 2.34324428327856]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.28193335474215, 2.56090122245649]  # 经过统计后的货车宽度范围