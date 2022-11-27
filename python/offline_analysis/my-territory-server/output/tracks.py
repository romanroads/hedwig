# !/usr/bin/python
# coding=utf-8

# TRACKS FILE
import sys

FRAME = 'frame'
ID = 'id'
X = 'x'
Y = 'y'
WIDTH = 'width'
HEIGHT = 'height'
X_VELOCITY = 'xVelocity'
Y_VELOCITY = 'yVelocity'
X_ACCELERATION = 'xAcceleration'
Y_ACCELERATION = 'yAcceleration'
FRONT_SIGHT_DISTANCE = 'frontSightDistance'
BACK_SIGHT_DISTANCE = 'backSightDistance'
DHW = 'dhw'
THW = 'thw'
TTC = 'ttc'
PRECEDING_X_VELOCITY = 'precedingXVelocity'
PRECEDING_ID = 'precedingId'
FOLLOWING_ID = 'followingId'
LEFT_PRECEDING_ID = 'leftPrecedingId'
LEFT_ALONGSIDE_ID = 'leftAlongsideId'
LEFT_FOLLOWING_ID = 'leftFollowingId'
RIGHT_PRECEDING_ID = 'rightPrecedingId'
RIGHT_ALONGSIDE_ID = 'rightAlongsideId'
RIGHT_FOLLOWING_ID = 'rightFollowingId'
LANE_ID = 'laneId'
X_TOP_LEFT_AABB = 'xTopLeftAABB'
Y_TOP_LEFT_AABB = 'yTopLeftAABB'
X_TOP_RIGHT_AABB = 'xTopRightAABB'
Y_TOP_RIGHT_AABB = 'yTopRightAABB'
X_BOTTOM_RIGHT_AABB = 'xBottomRightAABB'
Y_BOTTOM_RIGHT_AABB = 'yBottomRightAABB'
X_BOTTOM_LEFT_AABB = 'xBottomLeftAABB'
Y_BOTTOM_LEFT_AABB = 'yBottomLeftAABB'
X_TOP_LEFT_RBB = 'xTopLeftRBB'
Y_TOP_LEFT_RBB = 'yTopLeftRBB'
X_TOP_RIGHT_RBB = 'xTopRightRBB'
Y_TOP_RIGHT_RBB = 'yTopRightRBB'
X_BOTTOM_RIGHT_RBB = 'xBottomRightRBB'
Y_BOTTOM_RIGHT_RBB = 'yBottomRightRBB'
X_BOTTOM_LEFT_RBB = 'xBottomLeftRBB'
Y_BOTTOM_LEFT_RBB = 'yBottomLeftRBB'
RAW_WIDTH = 'rawWidth'
RAW_HEIGHT = 'rawHeight'
OLD_ID = 'oldId'
X_CENTER = 'xCenter'
Y_CENTER = 'yCenter'


def get_header():
    """
    获取csv的表头
    :return:
    """
    return [FRAME, ID, X, Y, WIDTH, HEIGHT, X_VELOCITY,
            Y_VELOCITY, X_ACCELERATION, Y_ACCELERATION, FRONT_SIGHT_DISTANCE,
            BACK_SIGHT_DISTANCE, DHW, THW, TTC, PRECEDING_X_VELOCITY,
            PRECEDING_ID, FOLLOWING_ID, LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID,
            LEFT_FOLLOWING_ID, RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID,
            RIGHT_FOLLOWING_ID, LANE_ID, X_TOP_LEFT_AABB, Y_TOP_LEFT_AABB,
            X_TOP_RIGHT_AABB, Y_TOP_RIGHT_AABB, X_BOTTOM_RIGHT_AABB,
            Y_BOTTOM_RIGHT_AABB, X_BOTTOM_LEFT_AABB, Y_BOTTOM_LEFT_AABB,
            X_TOP_LEFT_RBB, Y_TOP_LEFT_RBB,
            X_TOP_RIGHT_RBB, Y_TOP_RIGHT_RBB, X_BOTTOM_RIGHT_RBB,
            Y_BOTTOM_RIGHT_RBB, X_BOTTOM_LEFT_RBB, Y_BOTTOM_LEFT_RBB,
            RAW_WIDTH, RAW_HEIGHT, OLD_ID, X_CENTER, Y_CENTER]


class Track:
    frame = None
    id = None
    width = 0
    height = 0
    x = 0
    y = 0
    x_velocity = 0
    y_velocity = 0
    x_acceleration = 0
    y_acceleration = 0
    front_sight_distance = 0
    back_sight_distance = 0
    dhw = 0
    thw = 0
    ttc = 0
    preceding_id = None
    following_id = None
    left_preceding_id = None
    left_alongside_id = None
    left_following_id = None
    right_preceding_id = None
    right_alongside_id = None
    right_following_id = None
    preceding_x_velocity = 0
    lane_id = None
    up_or_down = None
    x_top_left_aabb = 0
    y_top_left_aabb = 0
    x_top_right_aabb = 0
    y_top_right_aabb = 0
    x_bottom_right_aabb = 0
    y_bottom_right_aabb = 0
    x_bottom_left_aabb = 0
    y_bottom_left_aabb = 0
    x_top_left_rbb = 0
    y_top_left_rbb = 0
    x_top_right_rbb = 0
    y_top_right_rbb = 0
    x_bottom_right_rbb = 0
    y_bottom_right_rbb = 0
    x_bottom_left_rbb = 0
    y_bottom_left_rbb = 0
    raw_width = 0
    raw_height = 0
    old_id = 0
    x_center = 0
    y_center = 0
    pet = sys.float_info.max
    lane_keep_intention = 'lane_keep'
    left_lane_change_intention = 'lane_keep'
    take_over_intention = 'not_taking_over'
    yaw = 0
    roll = 0
    pitch = 0
    rotation_speed = 0
    angular_velocity = rotation_speed
    steering_angle = 0

    def __str__(self):
        return ','.join(['%s' % item for item in self.__dict__.values()])


def read_row(row):
    track = Track()
    track.frame = row[0]
    track.id = row[1]
    track.x = row[2]
    track.y = row[3]
    track.width = row[4]
    track.height = row[5]
    track.x_velocity = row[6]
    track.y_velocity = row[7]
    track.x_acceleration = row[8]
    track.y_acceleration = row[9]
    track.front_sight_distance = row[10]
    track.back_sight_distance = row[11]
    track.dhw = row[12]
    track.thw = row[13]
    track.ttc = row[14]
    track.preceding_x_velocity = row[15]
    track.preceding_id = row[16]
    track.following_id = row[17]
    track.left_preceding_id = row[18]
    track.left_alongside_id = row[19]
    track.left_following_id = row[20]
    track.right_preceding_id = row[21]
    track.right_alongside_id = row[22]
    track.right_following_id = row[23]
    track.lane_id = row[24]
    track.x_top_left_aabb = row[25]
    track.y_top_left_aabb = row[26]
    track.x_top_right_aabb = row[27]
    track.y_top_right_aabb = row[28]
    track.x_bottom_right_aabb = row[29]
    track.y_bottom_right_aabb = row[30]
    track.x_bottom_left_aabb = row[31]
    track.y_bottom_left_aabb = row[32]

    return track
