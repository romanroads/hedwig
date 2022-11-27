import importlib

import cv2
import numpy as np
import os.path
import pandas

from util.lane_util import cale_angle


bg_setting = {
        "广州沈海高速(初采)": {
            "config_pre": "gzshgs_cc",
            "name": "广州沈海高速(初采)"
        },
        "广州沈海高速(补采)": {
            "config_pre": "gzshgs_bc",
            "name": "广州沈海高速(补采)"
        },
        "广州环城高速": {
            "config_pre": "gzhcgs",
            "name": "广州环城高速"
        },
        "上海沪昆高速": {
            "config_pre": "shhkgs",
            "name": "上海沪昆高速"
        },
        "上海绕城高速": {
            "config_pre": "shrcgs",
            "name": "上海绕城高速"
        },
        "广州广台高速": {
            "config_pre": "gzgtgs",
            "name": "广州广台高速"
        },
        "上海申嘉湖高速": {
            "config_pre": "shsjhgs",
            "name": "上海申嘉湖高速"
        }
    }


def draw_original(img_path, output, file_id, gc_background):
    # 读取图像
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # 绘制车道线
    for index, line in gc_background.default.items():
        # 线的起点和终点，线宽
        cv2.line(img, (0, line[0]), (3840, line[1]), color=(0, 0, 255), thickness=2)

    # 加标题文字
    # cv2.putText(img, str(file_id) + "_original", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    # 生成图像文件
    out_file_name = str(file_id) + "_original.jpg"
    cv2.imencode('.jpg', img)[1].tofile(output + out_file_name)


def draw_fix(img_path, output, file_id, gc_background):
    # 读取图像
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # 计算车道线倾斜角度
    angle_arr = []
    for index, line in gc_background.default.items():
        # 计算倾斜
        angle, cross_p = cale_angle(line)
        if angle > 180:
            angle = angle - 360
        angle_arr.append(angle)

    '''
    全部车道线倾斜度均值计算，并旋转图像
    '''
    # 平均倾斜度
    all_angle_avg = np.average(angle_arr)
    # 画面旋转
    all_rotated = rotated(img, all_angle_avg)

    '''
    上半部分车道线倾斜度均值计算，并旋转图像
    '''
    # 平均倾斜度
    up_angle_avg = np.average(angle_arr[0: 5])
    # 画面旋转
    up_rotated = rotated(img, up_angle_avg)

    '''
    下半部分车道线倾斜度均值计算，并旋转图像
    '''
    # 平均倾斜度
    down_angle_avg = np.average(angle_arr[5: ])
    # 画面旋转
    down_rotated = rotated(img, down_angle_avg)

    # 绘制车道线
    for index, line in gc_background.default.items():
        # 线的起点和终点，线宽
        cv2.line(all_rotated, (0, line[0]), (3840, line[0]), color=(0, 0, 255), thickness=2)
        cv2.line(up_rotated, (0, line[0]), (3840, line[0]), color=(0, 0, 255), thickness=2)
        cv2.line(down_rotated, (0, line[0]), (3840, line[0]), color=(0, 0, 255), thickness=2)

    # 加标题文字
    # cv2.putText(all_rotated, str(file_id) + "_fix_all", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    # cv2.putText(up_rotated, str(file_id) + "_fix_up", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    # cv2.putText(down_rotated, str(file_id) + "_fix_down", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    # 生成图像文件
    out_file_name = str(file_id) + "_fix_all.jpg"
    cv2.imencode('.jpg', all_rotated)[1].tofile(output + out_file_name)

    out_file_name = str(file_id) + "_fix_up.jpg"
    cv2.imencode('.jpg', up_rotated)[1].tofile(output + out_file_name)

    out_file_name = str(file_id) + "_fix_down.jpg"
    cv2.imencode('.jpg', down_rotated)[1].tofile(output + out_file_name)


def rotated(img, angle):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((0, 0), -angle, 1.0)
    return cv2.warpAffine(img, M, (0, h))


def main():

    summary_path = "F:/项目文档/绛门/陆领项目/recordingMeta_data_summary.csv"
    summary_pd = pandas.read_csv(summary_path)
    for index, row in summary_pd.iterrows():
        img_dir = "C:/Users/HUAWEI/Desktop/img_0328/"
        img_dir += row['locationName'] + "/"
        img_name = str(row['ID']) + ".jpg"
        img_path = img_dir + img_name
        print(img_path)
        if os.path.exists(img_path):
            # 道路背景设置
            pre = bg_setting.get(row['locationName'])
            if pre is None:
                continue

            if float(str(row['fileName']).split('.')[1]) == 0:
                suf = str(int(row['fileName']))
            else:
                suf = str(row['fileName']).replace('.', '_')
            bg = pre.get('config_pre') + '_' + suf
            module_name = 'background.' + bg
            try:
                gc_background = importlib.import_module(module_name)
            except:
                print("没有配置文件，文件名：{}".format(bg))
                continue

            draw_original(img_path, img_dir, row['ID'], gc_background)
            draw_fix(img_path, img_dir, row['ID'], gc_background)
    '''
    output = "C:/Users/HUAWEI/Desktop/img_0327/上海沪昆高速/"
    img_path = "C:\\Users\\HUAWEI\\Desktop\\img_0327\\上海沪昆高速\\42.jpg"
    draw_original(img_path, output, 42)
    draw_fix(img_path, output, 42)
    '''


if __name__ == '__main__':
    main()
