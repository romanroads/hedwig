# !/usr/bin/python
# coding=utf-8

class TracksMeta:
    """
    存储TrackMeta.csv文件的单条数据对象
    """

    def __init__(self, tracks_mate):
        """
        初始化方法
        :param tracks_mate: 封装的数据对象
        """
        self.id = tracks_mate["id"]
        self.width = tracks_mate["width"]
        self.height = tracks_mate["height"]
        self.initialFrame = tracks_mate["initialFrame"]
        self.finalFrame = tracks_mate["finalFrame"]
        self.numFrames = tracks_mate["numFrames"]
        self.classes = tracks_mate["classes"]
        self.drivingDirection = tracks_mate["drivingDirection"]
        self.traveledDistance = tracks_mate["traveledDistance"]
        self.minXVelocity = tracks_mate["minXVelocity"]
        self.maxXVelocity = tracks_mate["maxXVelocity"]
        self.meanXVelocity = tracks_mate["meanXVelocity"]
        self.minDHW = tracks_mate["minDHW"]
        self.minTHW = tracks_mate["minTHW"]
        self.minTTC = tracks_mate["minTTC"]
        self.numLaneChanges = tracks_mate["numLaneChanges"]

    @staticmethod
    def create_csv_title():
        """
        创建csv文件的表头
        :return: 表头集合数据
        """
        return ["id", "width", "height", "initialFrame", "finalFrame", "numFrames", "class",
                "drivingDirection", "traveledDistance", "minXVelocity", "maxXVelocity", "meanXVelocity",
                "minDHW", "minTHW", "minTTC", "numLaneChanges"]

    def case_to_list(self):
        """
        将对象转成序列化的集合数据
        :return: 转化后的集合
        """
        return [self.id, self.width, self.height, self.initialFrame, self.finalFrame, self.numFrames,
                self.classes, self.drivingDirection, self.traveledDistance, self.minXVelocity,
                self.maxXVelocity, self.meanXVelocity, self.minDHW, self.minTHW, self.minTTC,
                self.numLaneChanges]
