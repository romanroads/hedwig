import os
import logging
import numpy as np
import re
from pathlib import Path

from utils import get_video_file_artifact_file_name
from constant_values import *


class MinViableDataWriter(object):
    """
    Data schema of the output structured data

    CREATE TABLE IF NOT EXISTS min_viable_data_tracking
                    (user_id int,
                     video_id bigint,
                     exp bigint,
                     run bigint,
                     trip bigint,
                     id bigint,
                     frame_id bigint,
                     fractional_top_left_x real,
                     fractional_top_left_y real,
                     fractional_bottom_right_x real,
                     fractional_bottom_right_y real,
                     fractional_center_x real,
                     fractional_center_y real,
                     PRIMARY KEY(user_id, video_id, exp, run, trip, id, frame_id)
                    );
    """
    def __init__(self, video_file, last_commit_num_chunks, num_digits_for_print):
        _, self.csv_file_name = get_video_file_artifact_file_name(video_file, "_mvd.csv")

        dirname, _ = os.path.split(os.path.abspath(__file__))

        path_to_output = os.path.join(Path(dirname).parent.parent, PROCESSED_VIDEO_FOLDER_NAME)
        if os.path.exists(path_to_output) is False:
            os.mkdir(path_to_output)

        self.csv_file_path = path_to_output
        self.csv_file_changed = False
        self.csv_file_header_col_names = ["id", "frame_id",
                                          "fractional_top_left_x", "fractional_top_left_y", "fractional_bottom_right_x",
                                          "fractional_bottom_right_y", "fractional_center_x", "fractional_center_y"]

        self.regex = re.compile(r'(\d+|\s+)')
        self.list_query_rows = []
        self.last_commit_num_chunks = last_commit_num_chunks
        self.num_digits_for_print = num_digits_for_print
        self.format_string_float = f"%.{self.num_digits_for_print}f"

        # Note: these vectors are constant for aabb bounding box
        self.width_axis_vec = np.array([1., 0.])
        self.height_axis_vec = np.array([0., 1.])

    def write_a_row_for_tracklets(self, frame_number, data):

        if TRACKED_AGENTS_NAME not in data:
            return

        if DIMENSIONS_NAME not in data:
            return

        if ORI_FRAME_DIMEN_NAME not in data:
            return

        tracks_distributed_to_grids = data[TRACKED_AGENTS_NAME]

        width_authentic_resolution_roi, height_authentic_resolution_roi, \
        width_starting_point_authentic_resolution_roi, height_starting_point_authentic_resolution_roi, \
        width_tracking, height_tracking = data[DIMENSIONS_NAME]

        scale_tracking_to_ori_width = width_authentic_resolution_roi * 1. / width_tracking
        scale_tracking_to_ori_height = height_authentic_resolution_roi * 1. / height_tracking

        ori_image_width, ori_image_height = data[ORI_FRAME_DIMEN_NAME]

        for sublist in tracks_distributed_to_grids:
            for tracklet in sublist:

                if not tracklet.is_dynamic:
                    continue

                traj = tracklet.traj
                bbox = tracklet.bbox
                bbox_aabb = tracklet.bbox_aabb

                # Note: -1 index gives current position, most updated one
                pos = traj[-1]
                bound_box = bbox[-1]
                bound_box_aabb = bbox_aabb[-1]

                name = tracklet.name
                agent_type, agent_index, _ = self.regex.split(name)
                agent_id = tracklet.obj_id

                # Note: this is in tracking space
                x, y = pos
                poly_len, poly_width = bound_box  # rbb size, polygon length and width, large and small principal axis
                bbox_width_aabb, bbox_height_aabb = bound_box_aabb  # aabb size
                # Note: pytorch tensor to float variable
                bbox_width_aabb = bbox_width_aabb.item()
                bbox_height_aabb = bbox_height_aabb.item()

                # Note 4 corners of rbb bounding box
                l_axis_vec = tracklet.larger_principal_axis_vector
                s_axis_vec = tracklet.smaller_principal_axis_vector
                v_tl_rbb = pos + l_axis_vec * poly_len * 0.5 + s_axis_vec * poly_width * 0.5
                v_tr_rbb = pos - l_axis_vec * poly_len * 0.5 + s_axis_vec * poly_width * 0.5
                v_br_rbb = pos - l_axis_vec * poly_len * 0.5 - s_axis_vec * poly_width * 0.5
                v_bl_rbb = pos + l_axis_vec * poly_len * 0.5 - s_axis_vec * poly_width * 0.5
                x_tl_rbb, y_tl_rbb = v_tl_rbb
                x_tr_rbb, y_tr_rbb = v_tr_rbb
                x_br_rbb, y_br_rbb = v_br_rbb
                x_bl_rbb, y_bl_rbb = v_bl_rbb

                # Note 4 corners of aabb bounding box
                v_tl_aabb = pos - self.width_axis_vec * bbox_width_aabb * 0.5 - self.height_axis_vec * bbox_height_aabb * 0.5
                v_tr_aabb = pos + self.width_axis_vec * bbox_width_aabb * 0.5 - self.height_axis_vec * bbox_height_aabb * 0.5
                v_br_aabb = pos + self.width_axis_vec * bbox_width_aabb * 0.5 + self.height_axis_vec * bbox_height_aabb * 0.5
                v_bl_aabb = pos - self.width_axis_vec * bbox_width_aabb * 0.5 + self.height_axis_vec * bbox_height_aabb * 0.5
                x_tl_aabb, y_tl_aabb = v_tl_aabb
                x_tr_aabb, y_tr_aabb = v_tr_aabb
                x_br_aabb, y_br_aabb = v_br_aabb
                x_bl_aabb, y_bl_aabb = v_bl_aabb

                # Note: all coordinates below in units of fractional pixel coordinates
                x = self.get_frac_coordinates(x, scale_tracking_to_ori_width,
                                              width_starting_point_authentic_resolution_roi, ori_image_width)
                y = self.get_frac_coordinates(y, scale_tracking_to_ori_height,
                                              height_starting_point_authentic_resolution_roi, ori_image_height)

                x_tl_aabb = self.get_frac_coordinates(x_tl_aabb, scale_tracking_to_ori_width,
                                              width_starting_point_authentic_resolution_roi, ori_image_width)
                y_tl_aabb = self.get_frac_coordinates(y_tl_aabb, scale_tracking_to_ori_height,
                                              height_starting_point_authentic_resolution_roi, ori_image_height)

                x_br_aabb = self.get_frac_coordinates(x_br_aabb, scale_tracking_to_ori_width,
                                                      width_starting_point_authentic_resolution_roi, ori_image_width)
                y_br_aabb = self.get_frac_coordinates(y_br_aabb, scale_tracking_to_ori_height,
                                                      height_starting_point_authentic_resolution_roi, ori_image_height)

                self.list_query_rows.append((agent_id, frame_number, x_tl_aabb, y_tl_aabb, x_br_aabb, y_br_aabb,
                                             x, y
                                             ))

    @staticmethod
    def get_frac_coordinates(v, scale, offset, ori_dimen):
        return (v * scale * 1. + offset) / ori_dimen

    def transform_to_string_format_of_query_chunks(self):
        string_token_csv = ""

        num_rows = len(self.list_query_rows)
        interval_rows = int(num_rows / self.last_commit_num_chunks)
        row_index = 0
        chunk_index = 0

        logging.info("min_viable_data_writer: last commit with %s chunks %s rows %s interval row" %
                     (self.last_commit_num_chunks, interval_rows, num_rows))

        while row_index < num_rows:
            if interval_rows > 0 and row_index >= interval_rows * (chunk_index + 1):
                try:
                    if len(string_token_csv) > 0:
                        logging.info(
                            "min_viable_data_writer: last commit of chunk %s / %s, %s char query" %
                            (chunk_index, self.last_commit_num_chunks, len(string_token_csv)))

                        self.write_file(string_token_csv)
                except:
                    logging.info(
                        "min_viable_data_writer: last commit of chunk %s / %s, %s char query failed" %
                        (chunk_index, self.last_commit_num_chunks, len(string_token_csv)))

                string_token_csv = ""
                chunk_index += 1

            list_of_tokens = self.list_query_rows[row_index]

            token_content = ("%s,%s," +
                             self.format_string_float + "," + self.format_string_float + "," +
                             self.format_string_float + "," + self.format_string_float + "," +
                             self.format_string_float + "," + self.format_string_float) % list_of_tokens

            string_token_csv += token_content + "\n"

            row_index += 1

        # Note: remaining stuff....
        if len(string_token_csv) > 0:
            try:
                logging.info(
                    "min_viable_data_writer: remainder of last commit of chunk %s in addition to %s chunks, %s char query" %
                    (chunk_index, self.last_commit_num_chunks, len(string_token_csv)))

                self.write_file(string_token_csv)
            except:
                logging.info(
                    "min_viable_data_writer: remainder of last commit of chunk %s in addition to %s chunks, %s char query failed" %
                    (chunk_index, self.last_commit_num_chunks, len(string_token_csv)))

        self.list_query_rows.clear()

    def write_file(self, file_content):
        if not self.csv_file_changed:
            mode = "w"
            with open(os.path.join(self.csv_file_path, self.csv_file_name), mode) as f:
                header_line = ""
                for c in self.csv_file_header_col_names:
                    header_line += c + ","
                header_line = header_line[:-1] + "\n"
                f.write(header_line)
                f.write(file_content)
        else:
            mode = "a"
            with open(os.path.join(self.csv_file_path, self.csv_file_name), mode) as f:
                f.write(file_content)

        self.csv_file_changed = True

    def finish_up(self):
        if len(self.list_query_rows) > 0:
            self.transform_to_string_format_of_query_chunks()
        self.csv_file_changed = False

