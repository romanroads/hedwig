# !/usr/bin/python
# coding=utf-8

from shapely.geometry import Polygon
from config import default_config


class Lane:
    def __init__(self,
                 top_lane,
                 bottom_lane,
                 up_or_down,
                 is_near_mid,
                 angle,
                 cross_p,
                 pixel_ratio,
                 exclude_left_x):
        self.left_x = 0 + exclude_left_x
        self.right_x = default_config.width / pixel_ratio
        self.top_left_y = top_lane[0] / pixel_ratio
        self.top_right_y = top_lane[1] / pixel_ratio
        self.bottom_left_y = bottom_lane[0] / pixel_ratio
        self.bottom_right_y = bottom_lane[1] / pixel_ratio
        self.up_or_down = up_or_down
        self.is_near_mid = is_near_mid
        self.angle = angle
        self.cross_p = cross_p
        # 判断是上车道或下车道，上车道注册区在右边，销毁区在左边
        if up_or_down == 1:
            # 计算注册区坐标
            register_left_x = self.right_x - default_config.register_length / pixel_ratio
            register_top_y = self.top_left_y
            register_right_x = self.right_x
            register_bottom_y = self.bottom_left_y
            # 计算销毁区坐标
            destroy_top_x = self.left_x
            destroy_top_y = self.top_left_y
            destroy_bottom_x = self.left_x + default_config.register_length / pixel_ratio
            destroy_bottom_y = self.bottom_left_y
        # 上车道注册区在左边，销毁区在右边
        else:
            # 计算注册区坐标
            register_left_x = self.left_x
            register_top_y = self.top_left_y
            register_right_x = self.left_x + default_config.register_length / pixel_ratio
            register_bottom_y = self.bottom_left_y
            # 计算销毁区坐标
            destroy_top_x = self.right_x - default_config.register_length / pixel_ratio
            destroy_top_y = self.top_left_y
            destroy_bottom_x = self.right_x
            destroy_bottom_y = self.bottom_left_y
        # 生成车道polygon
        self.lane_polygon = Polygon([(self.left_x, self.top_left_y),
                                     (self.left_x, self.bottom_left_y),
                                     (self.right_x, self.bottom_right_y),
                                     (self.right_x, self.top_right_y)])
        # 生成校正后车道polygon
        self.fix_lane_polygon = Polygon([(self.left_x, self.top_left_y),
                                     (self.left_x, self.bottom_left_y),
                                     (self.right_x, self.bottom_left_y),
                                     (self.right_x, self.top_left_y)])
        # 生成注册区polygon
        self.register_polygon = Polygon([(register_left_x, register_top_y),
                                         (register_left_x, register_bottom_y),
                                         (register_right_x, register_bottom_y),
                                         (register_right_x, register_top_y)])
        # 生成销毁区polygon
        self.destroy_polygon = Polygon([(destroy_top_x, destroy_top_y),
                                        (destroy_top_x, destroy_bottom_y),
                                        (destroy_bottom_x, destroy_bottom_y),
                                        (destroy_bottom_x, destroy_top_y)])

    def __str__(self):
        return ','.join(['%s' % item for item in self.__dict__.values()])
