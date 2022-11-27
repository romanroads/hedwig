import os

import numpy as np
import pandas


count = 0

def calc_height(path):
    global count

    pd = pandas.read_csv(path)
    min_height = pd['height'].min()
    max_height = pd['height'].max()
    print("最小height：{}".format(min_height))
    print("最大height：{}".format(max_height))
    if min_height < 1.5:
        count += 1


def main():
    folder = "D:\\1\\0116\\output"
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.find('tracksMeta.csv') > 0:
                file_path = root + "\\" + file
                print("====================")
                print(file_path)
                calc_height(file_path)
    print("====================")
    print("小于1.5的视频数：{}".format(count))


if __name__ == '__main__':
    main()
