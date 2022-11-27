# !/usr/bin/python
# coding=utf-8

import argparse
import csv
import datetime
import importlib
import os
import datetime

import pandas

from background import bg_config
from entity.file import File
from util import result_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', required=True, help='mvd.csv folder')
    parser.add_argument('--rp', required=False, help='result file path')
    parser.add_argument('--sp', required=False, help='summary file path')
    parser.add_argument('--c', required=False, help='client id')

    # 开始时间
    start_time = datetime.datetime.now()

    # 命令参数
    opt = parser.parse_args()
    folder = opt.f
    result_path = opt.rp
    summary_path = opt.sp
    client_id = opt.c

    # 收集文件配置集合
    bg_setting_arr = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            root = root.replace("\\", "/")
            expressway_folder = root[root.rindex("/") + 1:]
            expressway_setting = bg_config.setting.get(expressway_folder)
            if expressway_setting:
                file_path = root + "/" + file
                video_id = file[:file.index("_")]
                file_config = expressway_setting.get("config_pre") + "_" + video_id
                file = File(video_id, file_config, file_path, expressway_setting.get("name"))
                bg_setting_arr.append(file)

    if bg_setting_arr:
        # 设置默认summary路径
        if not result_path:
            result_path = "/home/ec2-user/output/"
        print("文件输出路径：{}".format(result_path))

        # 自动创建目录
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # 设置默认summary路径
        if not summary_path:
            summary_path = "/home/ec2-user/recordingMeta_data_summary.csv"
        print("summary路径：{}".format(summary_path))

        # 需要处理的道路
        handle_data = []
        if os.path.exists(summary_path):
            summary_pd = pandas.read_csv(summary_path)
            for index, row in summary_pd.iterrows():
                if row['handle_flag'] == 1 and int(client_id) == row['client_id']:
                    handle_data.append(row)

        # 循环处理文件，并写入处理结果
        handle_result_path = result_path + "handle_result.csv"
        handle_result_open = open(handle_result_path, 'w', newline='')
        handle_result_writer = csv.writer(handle_result_open, dialect='excel')
        handle_result_writer.writerow(['expressway', 'video_id', 'bg_config', 'input_path', 'out_path'])
        for file_setting in bg_setting_arr:
            path = file_setting.path
            bg = file_setting.config_name
            result_type = "all"
            expressway_name = file_setting.expressway
            video_id = file_setting.video_id

            # 是否需要处理
            flag = False
            id = None
            for row in handle_data:
                if expressway_name == row['locationName'] and float(video_id) == row['fileName']:
                    flag = True
                    id = row['ID']
                    break

            # 不需要处理，则跳过
            if not flag:
                continue

            print("==========开始处理[{}]-[{}]数据===============".format(expressway_name, video_id))

            today = datetime.date.today()
            # 文件输出地址，比如 /output/01_xxx.csv
            output_path = result_path + str(id) + "_"

            # 道路背景设置
            bg = bg.replace(".", "_")
            module_name = 'background.' + bg
            try:
                gc_background = importlib.import_module(module_name)
            except:
                print("没有[{}]-[{}]的配置，配置文件名：{}".format(expressway_name, video_id, bg))
                continue

            # 生成结果
            result_util.generate(path, gc_background, result_type, output_path, expressway_name, video_id, summary_path)

            # 写入结果文件
            handle_result_writer.writerow([
                expressway_name,
                video_id,
                bg,
                path,
                output_path
            ])

    else:
        print("没有找到csv数据文件！")

    # 结束时间
    end_time = datetime.datetime.now()

    # 执行时间，单位分钟
    execution_time = (end_time - start_time).seconds / 60
    print("-----程序执行时间：{}".format(execution_time))


if __name__ == '__main__':
    main()
