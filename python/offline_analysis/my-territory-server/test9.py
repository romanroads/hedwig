import numpy as np
import pandas
from matplotlib import pyplot as plt
import statsmodels.api as sm


lowess = sm.nonparametric.lowess


def draw(o_data, l_data, label, name, path):
    print("开始生成" + name + "图片...")
    # 设置图片文件名
    img_name = name + ".png"
    img_path = path + img_name

    if o_data:
        plt.plot(o_data['x'], o_data['y'], label='Original', linewidth=0.5)
    if l_data:
        plt.plot(l_data['x'], l_data['y'], label='Lowess', linewidth=0.5)

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
    # x坐标平滑处理
    sm_x = lowess(data, np.arange(0, len(data), 1), frac=frac)
    filter_x_arr = sm_x[:, 1]

    # x速度计算
    old_x_v_arr = np.diff(filter_x_arr) * 30
    if len(old_x_v_arr) > 1:
        fisrt_x_velocity = old_x_v_arr[0] - (old_x_v_arr[1] - old_x_v_arr[0])
    else:
        fisrt_x_velocity = 0
    old_x_v_arr = np.insert(old_x_v_arr, 0, fisrt_x_velocity)  # 补充第一个x速度

    # x速度平滑处理
    sm_x_v = lowess(old_x_v_arr, np.arange(0, len(old_x_v_arr), 1), frac=frac)
    filter_x_v_arr = sm_x_v[:, 1]

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


def handle2(data, frac):
    # x速度计算
    x_v_arr = np.diff(data)
    if len(x_v_arr) > 1:
        fisrt_x_velocity = x_v_arr[0] - (x_v_arr[1] - x_v_arr[0])
    else:
        fisrt_x_velocity = 0
    x_v_arr = np.insert(x_v_arr, 0, fisrt_x_velocity)  # 补充第一个x速度
    sm_x_a = lowess(x_v_arr, np.arange(0, len(x_v_arr), 1), frac=frac)
    filter_x_a_arr = sm_x_a[:, 1]

    # x加速度计算
    x_a_v_arr = np.diff(filter_x_a_arr)
    if len(x_a_v_arr) > 1:
        fisrt_x_a_velocity = x_a_v_arr[0] - (x_a_v_arr[1] - x_a_v_arr[0])
    else:
        fisrt_x_a_velocity = 0
    x_a_v_arr = np.insert(x_a_v_arr, 0, fisrt_x_a_velocity)  # 补充第一个x加速度

    # x加速度平滑处理
    sm_x_a_v = lowess(x_a_v_arr, np.arange(0, len(x_a_v_arr), 1), frac=frac)
    filter_x_a_v_arr = sm_x_a_v[:, 1]
    return x_a_v_arr, filter_x_a_v_arr, filter_x_a_arr

def main():
    in_path = "F:\\项目文档\\绛门\\陆领项目\\img1220\\original_tracks.csv"
    out_path = "C:\\Users\\HUAWEI\\Desktop\\7\\"
    pd = pandas.read_csv(in_path)
    frame_arr, x_arr, y_arr = [], [], []
    for index, row in pd.iterrows():
        if row['id'] != 6:
           continue
        frame = row['frame']
        fractional_center_x = row['x']
        fractional_center_y = row['y']
        frame_arr.append(frame)
        x_arr.append(fractional_center_x)
        y_arr.append(fractional_center_y)

    # 生成图片
    frac = round((30 / len(frame_arr)), 2)
    # frac = 0.01
    old_x_v_arr, old_x_a_v_arr = old_handle(x_arr)
    new_x_arr, new_x_v_arr, new_x_a_v_arr = handle1(x_arr, frac)
    o_data = {'x': frame_arr, 'y': old_x_a_v_arr}
    l_data = {'x': frame_arr, 'y': new_x_a_v_arr}
    #x_data = {'x': frame_arr, 'y': lowess_x_arr * 30}
    label = {'x': 'Frame ID', 'y': 'X_A_V'}

    draw(o_data, l_data, label, 'X_A_V1 (' + str(frac) + ')', out_path)
    draw(None, l_data, label, 'X_A_V2 (' + str(frac) + ')', out_path)

    # X对比
    o_data = {'x': frame_arr, 'y': x_arr}
    l_data = {'x': frame_arr, 'y': new_x_arr}
    label = {'x': 'Frame ID', 'y': 'X Center'}

    draw(o_data, l_data, label, 'X_Center1 (' + str(frac) + ')', out_path)
    draw(None, l_data, label, 'X_Center2 (' + str(frac) + ')', out_path)

    # 速度对比
    o_data = {'x': frame_arr, 'y': old_x_v_arr}
    l_data = {'x': frame_arr, 'y': new_x_v_arr}
    label = {'x': 'Frame ID', 'y': 'X_V'}

    draw(o_data, l_data, label, 'X_V1 (' + str(frac) + ')', out_path)
    draw(None, l_data, label, 'X_V2 (' + str(frac) + ')', out_path)

    # 加速度差值
    # x加速度计算
    before_x_a_v_arr = np.diff(new_x_v_arr) * 30
    if len(before_x_a_v_arr) > 1:
        fisrt_x_a_velocity = before_x_a_v_arr[0] - (before_x_a_v_arr[1] - before_x_a_v_arr[0])
    else:
        fisrt_x_a_velocity = 0
    before_x_a_v_arr = np.insert(before_x_a_v_arr, 0, fisrt_x_a_velocity)  # 补充第一个x加速度
    x_a_v_diff = new_x_a_v_arr - before_x_a_v_arr
    print("x_a_v_diff：{}".format(sum(x_a_v_diff)))
    l_data = {'x': frame_arr, 'y': x_a_v_diff}
    label = {'x': 'Frame ID', 'y': 'X_A_V_Diff'}
    draw(None, l_data, label, 'X_A_V_Diff2 (' + str(frac) + ')', out_path)

if __name__ == '__main__':
    main()
