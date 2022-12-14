# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(补采)-21）
"""
default = {
    1: [944, 942],
    2: [970, 966],
    3: [1001, 997],
    4: [1033, 1028],
    5: [1064, 1058],
    6: [1088, 1078],
    7: [1118, 1108],
    8: [1150, 1140],
    9: [1182, 1170],
    10: [1208, 1196]
}
pixel_ratio = 8.27  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.36523992346591, 2.1872760043445]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.15287226847207, 2.6503218467822]  # 经过统计后的货车宽度范围
