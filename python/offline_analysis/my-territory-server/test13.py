import csv
import os

import numpy as np
import pandas
from matplotlib import pyplot as plt
import statsmodels.api as sm


lowess = sm.nonparametric.lowess


def draw(o_data, l_data, c_data, label, name, path):
    print("开始生成" + name + "图片...")
    # 设置图片文件名
    img_name = name + ".png"
    img_path = path + img_name

    # 自动创建目录
    if not os.path.exists(path):
        os.makedirs(path)

    if o_data:
        plt.plot(o_data['x'], o_data['y'], label=o_data['l'], linewidth=0.5)
    if l_data:
        plt.plot(l_data['x'], l_data['y'], label=l_data['l'], linewidth=0.5)
    if c_data:
        plt.plot(l_data['x'], c_data['y'], label=c_data['l'], linewidth=0.5)

    plt.xlabel(label['x'])
    plt.ylabel(label['y'])
    plt.gca().xaxis.set_ticks_position('top')
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print(name + "图片生成完毕！")


def old_handle(data):
    # x速度计算
    x_v_arr = np.diff(data) * 30
    if len(x_v_arr) > 1:
        fisrt_x_velocity = x_v_arr[0] - (x_v_arr[1] - x_v_arr[0])
    else:
        fisrt_x_velocity = 0
    x_v_arr = np.insert(x_v_arr, 0, fisrt_x_velocity)  # 补充第一个x速度

    # x加速度计算
    x_a_v_arr = np.diff(x_v_arr) * 30
    if len(x_a_v_arr) > 1:
        fisrt_x_a_velocity = x_a_v_arr[0] - (x_a_v_arr[1] - x_a_v_arr[0])
    else:
        fisrt_x_a_velocity = 0
    x_a_v_arr = np.insert(x_a_v_arr, 0, fisrt_x_a_velocity)  # 补充第一个x加速度

    return x_v_arr, x_a_v_arr


def handle1(data, frac):
    filter_x_arr = data

    # x速度计算
    old_x_v_arr = np.diff(filter_x_arr) * 30
    if len(old_x_v_arr) > 1:
        fisrt_x_velocity = old_x_v_arr[0] - (old_x_v_arr[1] - old_x_v_arr[0])
    else:
        fisrt_x_velocity = 0
    filter_x_v_arr = np.insert(old_x_v_arr, 0, fisrt_x_velocity)  # 补充第一个x速度

    # x加速度计算
    old_x_a_v_arr = np.diff(filter_x_v_arr) * 30
    if len(old_x_a_v_arr) > 1:
        fisrt_x_a_velocity = old_x_a_v_arr[0] - (old_x_a_v_arr[1] - old_x_a_v_arr[0])
    else:
        fisrt_x_a_velocity = 0
    old_x_a_v_arr = np.insert(old_x_a_v_arr, 0, fisrt_x_a_velocity)  # 补充第一个x加速度

    # x加速度平滑处理
    sm_x_a_v = lowess(old_x_a_v_arr, np.arange(0, len(old_x_a_v_arr), 1), frac=frac)
    filter_x_a_v_arr = sm_x_a_v[:, 1]
    return filter_x_arr, filter_x_v_arr, filter_x_a_v_arr


def calc_handle(filter_x_a_v_arr, f_x, f_x_v):
    filter_x_a_v_arr = filter_x_a_v_arr / 30

    # 重新计算x速度
    x_arr = [f_x]
    x_v_arr = [f_x_v]
    x_a_v_arr = [filter_x_a_v_arr[0]]
    index = 0
    for x_a_v in filter_x_a_v_arr:
        if index > 0:
            cur_x_v = x_v_arr[-1] + x_a_v
            x_v_arr.append(cur_x_v)
            cur_x = x_arr[-1] + cur_x_v / 30
            x_arr.append(cur_x)
            x_a_v_arr.append(x_a_v)
        index += 1
    # x_a_v_arr = np.array(x_a_v_arr) * 30

    return x_arr, x_v_arr, x_a_v_arr


def main():
    in_path = "F:\\项目文档\\绛门\\陆领项目\\0315\\original_tracks.csv"
    out_path = "F:\\项目文档\\绛门\\陆领项目\\0315\\data\\2次变道\\41\\"
    pd = pandas.read_csv(in_path)
    frame_arr, x_arr, y_arr = [], [], []
    for index, row in pd.iterrows():
        if row['id'] != 41:
            continue
        frame = row['frame']
        fractional_center_x = row['x']
        fractional_center_y = row['y']
        # frame = row['frame_id']
        # fractional_center_x = row['fractional_center_x']
        # fractional_center_y = row['fractional_center_y']
        frame_arr.append(frame)
        x_arr.append(fractional_center_x)
        y_arr.append(fractional_center_y)

    # 生成图片
    frac = round((30 / len(frame_arr)), 5)
    #frac = 0.12
    # 计算x
    old_x_v_arr, old_x_a_v_arr = old_handle(x_arr)
    new_x_arr, new_x_v_arr, new_x_a_v_arr = handle1(x_arr, frac)
    # 计算y
    old_y_v_arr, old_y_a_v_arr = old_handle(y_arr)
    new_y_arr, new_y_v_arr, new_y_a_v_arr = handle1(y_arr, frac)


    # XY对比
    o_data = {'x': x_arr, 'y': y_arr, 'l': 'Original'}
    l_data = {'x': new_x_arr, 'y': new_y_arr, 'l': 'Lowess'}
    label = {'x': 'X Center', 'y': 'Y Center'}

    draw(o_data, l_data, None, label, 'XY_1 (' + str(frac) + ')', out_path)

    # X速度对比
    o_data = {'x': x_arr, 'y': old_x_v_arr, 'l': 'Original'}
    l_data = {'x': new_x_arr, 'y': new_x_v_arr, 'l': 'Lowess'}
    label = {'x': 'X Center', 'y': 'X_V'}

    draw(o_data, l_data, None, label, 'X_V_1 (' + str(frac) + ')', out_path)

    # X加速度对比
    o_data = {'x': x_arr, 'y': old_x_a_v_arr, 'l': 'Original'}
    l_data = {'x': new_x_arr, 'y': new_x_a_v_arr, 'l': 'Lowess'}
    label = {'x': 'X Center', 'y': 'X_A_V'}

    draw(o_data, l_data, None, label, 'X_A_V_1 (' + str(frac) + ')', out_path)

    # Y速度对比
    o_data = {'x': x_arr, 'y': old_y_v_arr, 'l': 'Original'}
    l_data = {'x': new_x_arr, 'y': new_y_v_arr, 'l': 'Lowess'}
    label = {'x': 'X Center', 'y': 'Y_V'}

    draw(o_data, l_data, None, label, 'Y_V_1 (' + str(frac) + ')', out_path)

    # Y加速度对比
    o_data = {'x': x_arr, 'y': old_y_a_v_arr, 'l': 'Original'}
    l_data = {'x': new_x_arr, 'y': new_y_a_v_arr, 'l': 'Lowess'}
    label = {'x': 'X Center', 'y': 'Y_A_V'}

    draw(o_data, l_data, None, label, 'Y_A_V_1 (' + str(frac) + ')', out_path)

    """
    csv_path = "C:\\Users\\HUAWEI\\Desktop\\0215-1\\1次变道\\34\\y_a_v.csv"
    file_open = open(csv_path, 'w', newline='')
    file = csv.writer(file_open, dialect='excel')
    for y_a_v in new_y_a_v_arr:
        file.writerow([y_a_v])
    """


if __name__ == '__main__':
    main()
