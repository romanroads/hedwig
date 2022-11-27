import os
import optparse
import sys
import pandas as pd
import numpy as np
import random
from scipy import stats

sys.path.insert(0, '..')

from constant_values import *


def add_flag_to_processed_csv(data_folder):
    text_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Note: for text files, there are three kinds of them, the annotation ones, and the ROI ones, the backup
            # files which come from backward compatibility
            if file.endswith("_mvd.csv"):
                text_files.append(os.path.join(root, file))

    failure_rate_threshold = 0.05

    start_time = 3.
    start_time_to_end = 20.

    fps = 30.

    for text_file in text_files:
        df = pd.read_csv(text_file)
        ids = df.id.unique()
        frame_ids = df.frame_id.unique()
        min_frame_id = min(frame_ids)
        max_frame_id = max(frame_ids)
        max_time = max_frame_id / fps
        num_agents = len(ids)

        print("we looking at file %s, %d agents, min %d max %d frame, max time %.1f [min]" %
              (text_file, num_agents, min_frame_id, max_frame_id, (max_time / 60.)))

        frame_max = int(np.clip((max_time - start_time_to_end) * fps, 0, np.finfo(float).max))
        frame_min = int(start_time * fps)

        print("we are looking for spawning frame between min %d and max %d" % (frame_min, frame_max))

        distances = []
        agent_ids = []

        for id in ids:
            df_id = df.loc[df.id == id]
            start_frame_id = df_id.frame_id.min()
            end_frame_id = df_id.frame_id.max()

            if not (frame_min <= start_frame_id <= frame_max):
                continue

            start_x = df_id.loc[df_id.frame_id == start_frame_id].fractional_center_x.values[0]
            end_x = df_id.loc[df_id.frame_id == end_frame_id].fractional_center_x.values[0]
            distance_x = np.clip(np.abs(start_x - end_x), 0, 1)
            distances.append(distance_x)
            agent_ids.append(id)

        ave_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        sigma = np.sqrt(stats.moment(distances, moment=2))

        threshold_dist = ave_dist - sigma
        threshold_min = 0.2

        print("we see average dist %.3f, min dist %.3f max dist %.3f, with sigma %.3f, set threshold at %.3f"
              % (ave_dist, min_dist, max_dist, sigma, threshold_dist))

        distances_arr = np.array(distances)
        indicies = np.where((threshold_min < distances_arr) & (distances_arr < threshold_dist))
        indicies = indicies[0]
        agent_ids_arr = np.array(agent_ids)
        miss_ids = agent_ids_arr[indicies]

        indicies = np.where(distances_arr > ave_dist)
        indicies = indicies[0]
        success_ids = agent_ids_arr[indicies]

        print("we found %d missing agents: their ids are %s" % (len(miss_ids), miss_ids))

        num_miss = len(miss_ids)
        num_success = len(success_ids)
        num_total = num_miss + num_success

        failure_rate = num_miss * 1. / num_total
        success_rate = num_success * 1. / num_total

        print("we measure failure rate %.3f%%, success rate %.3f%%" % (failure_rate * 100., success_rate * 100.))

        if failure_rate > failure_rate_threshold:
            print("the measured failure rate is larger than the threshold of %.3f%%" % (failure_rate_threshold * 100.))
            failure_rate = failure_rate_threshold

        num_miss = int(num_success * failure_rate)
        ids_miss = random.choices(population=miss_ids, k=num_miss)
        print("we decided to select %d miss agents with respect to %d success agents, "
              "their ids are %s" % (num_miss, num_success, ids_miss))

        df['tag'] = False
        for id in ids_miss:
            df.loc[df.id == id, "tag"] = True

        print(df)

        file_name_out = os.path.join(".", os.path.basename(text_file))

        print(file_name_out)

        df.to_csv(file_name_out, index=False)


def main():
    parser = optparse.OptionParser()
    parser.add_option('--csv_folder', action="store", default="./", help=None)

    options, args = parser.parse_args()
    add_flag_to_processed_csv(options.csv_folder)


if __name__ == "__main__":
    main()
