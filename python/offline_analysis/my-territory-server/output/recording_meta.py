# !/usr/bin/python
# coding=utf-8

class RecordingMeta:
    """
    recordingMeta.csv文件的对象数据
    """

    speedLimit = 1

    def __init__(self, recording_meta):
        """
        初始化
        :type recording_meta: object recording_meta单个对象的字典数据
        """
        self.id = recording_meta["id"]
        self.frameRate = recording_meta["frameRate"]
        self.locationId = recording_meta["locationId"]
        self.speedLimit = recording_meta["speedLimit"]
        self.month = recording_meta["month"]
        self.weekDay = recording_meta["weekDay"]
        self.startTime = recording_meta["startTime"]
        self.duration = recording_meta["duration"]
        self.totalDrivenDistance = recording_meta["totalDrivenDistance"]
        self.totalDrivenTime = recording_meta["totalDrivenTime"]
        self.numVehicles = recording_meta["numVehicles"]
        self.numCars = recording_meta["numCars"]
        self.numTrucks = recording_meta["numTrucks"]
        self.upperLaneMarkings = recording_meta["upperLaneMarkings"]
        self.lowerLaneMarkings = recording_meta["lowerLaneMarkings"]
        self.frameWidth = recording_meta["frameWidth"]
        self.frameHeight = recording_meta["frameHeight"]

    @staticmethod
    def create_csv_title():
        """
        创建csv文件的表头
        :return: 表头集合数据
        """
        return ["id", "frameRate", "locationId", "speedLimit", "month", "weekDay", "startTime",
                "duration", "totalDrivenDistance", "totalDrivenTime", "numVehicles", "numCars",
                "numTrucks", "upperLaneMarkings", "lowerLaneMarkings", "frameWidth", "frameHeight"]

    def case_to_list(self):
        """
        将对象转成序列化的集合数据
        :return: 转化后的集合
        """
        return [self.id, self.frameRate, self.locationId, self.speedLimit, self.month, self.weekDay, self.startTime,
                self.duration, self.totalDrivenDistance, self.totalDrivenTime, self.numVehicles,
                self.numCars, self.numTrucks, self.upperLaneMarkings, self.lowerLaneMarkings, self.frameWidth,
                self.frameHeight]
