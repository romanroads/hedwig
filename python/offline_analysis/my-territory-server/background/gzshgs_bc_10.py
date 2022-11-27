# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(补采)-10）
"""
default = {
    1: [962, 943],
    2: [989, 969],
    3: [1021, 1000],
    4: [1054, 1033],
    5: [1086, 1063],
    6: [1110, 1084],
    7: [1142, 1116],
    8: [1174, 1148],
    9: [1206, 1180],
    10: [1233, 1206]
}
pixel_ratio = 8.53  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.4354844807697, 2.35962740959441]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.11839844057907, 2.68151832843913]  # 经过统计后的货车宽度范围
