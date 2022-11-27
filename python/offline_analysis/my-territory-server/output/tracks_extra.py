# !/usr/bin/python
# coding=utf-8

# TRACKS EXTRA FILE
FRAME = 'frame'
ID = 'id'
PET = 'PET'
LANE_KEEP_INTENTION = 'background-keep intention'
LEFT_LANE_CHANGE_INTENTION = 'left background change intention'
TAKE_OVER_INTENTION = 'take over intention'
VEHICLE_YAW = 'vehicle yaw'
VEHICLE_ROLL = 'vehicle roll'
VEHICLE_PITCH = 'vehicle pitch'
ROTATION_SPEED = 'rotation speed'
ANGULAR_VELOCITY = 'angular velocity'
STEERING_ANGLE = 'steering angle'


def get_header():
    """
    获取csv的表头
    :return:
    """
    return [FRAME, ID,
            PET, LANE_KEEP_INTENTION, LEFT_LANE_CHANGE_INTENTION,
            TAKE_OVER_INTENTION, VEHICLE_YAW, VEHICLE_ROLL,
            VEHICLE_PITCH, ROTATION_SPEED, ANGULAR_VELOCITY,
            STEERING_ANGLE]
