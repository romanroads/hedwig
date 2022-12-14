# !/usr/bin/python
# coding=utf-8

"""
道路配置参数（上海沪昆高速-9）
"""
default = {
    1: [954, 929],
    2: [977, 950],
    3: [1008, 982],
    4: [1041, 1015],
    5: [1072, 1047],
    6: [1107, 1082],
    7: [1138, 1112],
    8: [1169, 1144],
    9: [1200, 1177],
    10: [1225, 1201]
}
pixel_ratio = 8.53  # 画面像素和实际单位的比例，像素:米的比例
exclude_left_x = 0  # 排除左边的x左边值，单位米
statistics_car_height_range = [1.43198265842876, 2.34415042050847]  # 经过统计后的轿车宽度范围
statistics_truck_height_range = [2.25501149426324, 3.00258324711277]  # 经过统计后的货车宽度范围
