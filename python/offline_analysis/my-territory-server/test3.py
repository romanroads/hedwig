import numpy as np
import pandas
from matplotlib import pyplot as plt


def distribution(id_data, data):
    # 最大值
    max_val = round(max(data), 2)
    min_val = round(min(data), 2)

    # 数值差
    interval_arr = np.arange(min_val, max_val + 0.01, 0.01)
    interval_len = len(interval_arr)

    intervals = {}
    for i in range(interval_len):
        if i == interval_len - 1:
            break
        key = str(round(interval_arr[i], 2)) + '~' + str(round(interval_arr[i + 1], 2))
        intervals.update({key: []})

    print(intervals)

    for i, d in enumerate(data):
        for interval in intervals:
            start, end = tuple(interval.split('~'))
            if float(start) <= d <= float(end):
                intervals[interval].append(id_data[i])

    for i in intervals:
        count = len(intervals[i])
        rate = str(round((count / len(data)) * 100, 4)) + '%'
        cars = len(set(intervals[i]))

        print(i, count, rate, cars)

    """
    x = list(intervals.keys())
    y = list(intervals.values())

    plt.plot(x, y, linewidth=0.5)
    plt.xlabel("Interval")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    # plt.savefig(img_path)
    plt.close()
    print("区间分布图生成完毕！")
    """

def main():
    path = "D:\\1\\output\\1_tracks.csv"
    pd = pandas.read_csv(path)
    id_data = pd['oldId'].values
    data = pd['yAcceleration'].values
    distribution(id_data, data)


if __name__ == '__main__':
    main()
