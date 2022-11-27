# 轨迹集合
tracks_dict = {}

# 最大帧数
max_frame = 0

# ID对应字典
id_dict = {}


def pre_filter():
    global tracks_dict
    global max_frame
    global id_dict

    tracks_dict.update({1: 'a1'})
    id_dict.update({999: 'a999'})
    max_frame = 10000
