import codecs
import csv
import os

import numpy as np
import pandas


def calc_height(path):
    pd = pandas.read_csv(path)
    car_height_data = pd[pd['class'] == 'Car']['height']
    car_height_min = car_height_data.nsmallest(5).mean()
    car_height_max = car_height_data.nlargest(5).mean()
    print("Car最小height：{}".format(car_height_min))
    print("Car最大height：{}".format(car_height_max))
    print("----------")
    truck_height_data = pd[pd['class'] == 'Truck']['height']
    truck_height_min = truck_height_data.nsmallest(5).mean()
    truck_height_max = truck_height_data.nlargest(5).mean()
    print("Truck最小height：{}".format(truck_height_min))
    print("Truck最大height：{}".format(truck_height_max))

    data = {
        "car_height_min": car_height_min,
        "car_height_max": car_height_max,
        "truck_height_min": truck_height_min,
        "truck_height_max": truck_height_max
    }

    return data

def main():
    folder = "D:\\1\\0116\\output"
    out_path = "D:\\1\\0116\\height_statistics.csv"
    summary_path = "F:\\项目文档\\绛门\\陆领项目\\recordingMeta_data_summary.csv"
    pd = pandas.read_csv(summary_path)

    # 写入结果文件
    out_header = ['id', 'locationName', 'fileName', 'car_height_min', 'car_height_max', 'truck_height_min', 'truck_height_max']
    statistics_open = codecs.open(out_path, 'w', 'gbk')
    statistics_writer = csv.writer(statistics_open, dialect='excel')
    statistics_writer.writerow(out_header)

    # 数据文件处理
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.find('tracksMeta.csv') > 0:
                id = file[:file.index("_")]
                file_path = root + "\\" + file
                print("====================")
                print(file_path)
                # 统计宽度
                height_data = calc_height(file_path)
                # 写入结果
                summary_data = pd[pd['ID'] == int(id)]
                out_data = [
                    summary_data['ID'].values[0],
                    summary_data['locationName'].values[0],
                    summary_data['fileName'].values[0],
                    height_data['car_height_min'],
                    height_data['car_height_max'],
                    height_data['truck_height_min'],
                    height_data['truck_height_max']
                ]
                statistics_writer.writerow(out_data)


if __name__ == '__main__':
    main()
