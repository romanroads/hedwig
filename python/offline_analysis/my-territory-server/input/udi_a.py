# !/usr/bin/python
# coding=utf-8

# UDI A FILE
ID = "id"
FRAME_ID = "frame_id"
FRACTIONAL_TOP_LEFT_X = "fractional_top_left_x"
FRACTIONAL_TOP_LEFT_Y = "fractional_top_left_y"
FRACTIONAL_BOTTOM_RIGHT_X = "fractional_bottom_right_x"
FRACTIONAL_BOTTOM_RIGHT_Y = "fractional_bottom_right_y"
FRACTIONAL_CENTER_X = "fractional_center_x"
FRACTIONAL_CENTER_Y = "fractional_center_y"
LANE_ID = "lane_id"
TAG = "tag"
STATUS = "status"
MESSAGE = "message"


class InputEntity:
    def __init__(self,
                 id,
                 old_id,
                 frame_id,
                 fractional_top_left_x,
                 fractional_top_left_y,
                 fractional_bottom_right_x,
                 fractional_bottom_right_y,
                 fractional_center_x,
                 fractional_center_y,
                 x_velocity,
                 y_velocity,
                 x_a_velocity,
                 y_a_velocity,
                 width,
                 height,
                 road_id,
                 up_or_down,
                 is_in_destroy):
        self.id = id
        self.old_id = old_id
        self.frame_id = frame_id
        self.fractional_top_left_x = fractional_top_left_x
        self.fractional_top_left_y = fractional_top_left_y
        self.fractional_bottom_right_x = fractional_bottom_right_x
        self.fractional_bottom_right_y = fractional_bottom_right_y
        self.fractional_center_x = fractional_center_x
        self.fractional_center_y = fractional_center_y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.x_a_velocity = x_a_velocity
        self.y_a_velocity = y_a_velocity
        self.width = width
        self.height = height
        self.road_id = road_id
        self.up_or_down = up_or_down
        self.is_in_destroy = is_in_destroy

    def __str__(self):
        return ','.join(['%s' % item for item in self.__dict__.values()])
