# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（广州沈海高速(补采)-19）
"""
default = {
    1: [930, 966],
    2: [957, 989],
    3: [990, 1021],
    4: [1022, 1052],
    5: [1055, 1082],
    6: [1079, 1102],
    7: [1112, 1134],
    8: [1144, 1166],
    9: [1176, 1197],
    10: [1203, 1222]
}
pixel_ratio = 8.27  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 30  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.43220142698992, 2.39473060108138]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.13238080061976, 2.55577302203197]  # 经过统计后的货车宽度范围