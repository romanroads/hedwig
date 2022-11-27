import timeit

from pipeline.pipeline import Pipeline
from common.min_viable_data_writer import MinViableDataWriter
from constant_values import *


class UploadToCloud(Pipeline):
    def __init__(self, exp, run_number, trip, tag, fps, commit, cloud_country_code, auto_processing, dump_csv,
                 commit_user_req_data, commit_min_viable_data, is_velo_sample_slow_down, velo_sample_rate,
                 is_acc_sample_slow_down, acc_sample_rate, is_fit_req_to_speed, speed_poly_fit_order,
                 is_periodically_upload_user_data, upload_interval_user_data, num_digits_for_print,
                 last_commit_num_chunks, constrain_x_y_into_lane_polygon, is_to_removed_bad_quality_tracklets,
                 bad_quality_tracklets_dist, user_id=0, license_id="na"):
        self.exp = exp
        self.run_number = run_number
        self.trip = trip
        self.tag = tag
        self.fps = fps
        self.user_id = user_id
        self.license_id = license_id
        self.commit = commit
        self.cloud_country_code = cloud_country_code

        self.path_to_saved_video = None
        self.key_of_video_file_for_cloud = None
        self.is_path_to_local_and_cloud_processed_video_set = False
        self.auto_processing = auto_processing
        self.dump_csv = dump_csv

        self.commit_user_req_data = commit_user_req_data
        self.commit_min_viable_data = commit_min_viable_data
        self.is_velo_sample_slow_down = is_velo_sample_slow_down
        self.velo_sample_rate = velo_sample_rate
        self.is_acc_sample_slow_down = is_acc_sample_slow_down
        self.acc_sample_rate = acc_sample_rate
        self.is_fit_req_to_speed = is_fit_req_to_speed
        self.speed_poly_fit_order = speed_poly_fit_order

        self.is_periodically_upload_user_data = is_periodically_upload_user_data
        self.upload_interval_user_data = upload_interval_user_data
        self.num_digits_for_print = num_digits_for_print
        self.last_commit_num_chunks = last_commit_num_chunks
        self.constrain_x_y_into_lane_polygon = constrain_x_y_into_lane_polygon
        self.is_to_removed_bad_quality_tracklets = is_to_removed_bad_quality_tracklets
        self.bad_quality_tracklets_dist = bad_quality_tracklets_dist

        self.frame_number = 0  # always start from 0, even when we start from middle of a video file

        self.min_viable_data_writer = None

        super().__init__("UploadToCloud")

    def map(self, data):
        start_time = timeit.default_timer()

        if not self.is_path_to_local_and_cloud_processed_video_set:
            if SAVED_VIDEO_FILE_CLOUD_KEY in data and SAVED_VIDEO_FILE_PATH in data:
                self.key_of_video_file_for_cloud = data[SAVED_VIDEO_FILE_CLOUD_KEY]
                self.path_to_saved_video = data[SAVED_VIDEO_FILE_PATH]
                self.is_path_to_local_and_cloud_processed_video_set = True

        # TODO needs to move this block below into user req data writer block....
        corr_rot, corr_x, corr_y, updated_lane_polygons = 0., 0., 0., []
        if STABLIZATION_SYSTEM_NAME in data:
            stablization = data[STABLIZATION_SYSTEM_NAME]
            # Note: x and y shift are pixels in tracking image space
            corr_rot = stablization.correction_rot_angle
            corr_x = stablization.correction_shift_width
            corr_y = stablization.correction_shift_height

            for lane_index in range(len(stablization.config_lane_slots)):
                # Note: here the polygon point coordinates are in original image space
                lane_slot = stablization.config_lane_slots[lane_index]
                updated_lane_polygons.append(lane_slot)

        if self.commit_min_viable_data and self.min_viable_data_writer is None:
            videl_file_name = video_name = data[VIDEO_NAME]
            self.min_viable_data_writer = MinViableDataWriter(videl_file_name, self.last_commit_num_chunks,
                                                              self.num_digits_for_print)

        frame_number = data[FRAME_NUMBER_NAME]
        self.frame_number = frame_number

        width_authentic_resolution_roi, height_authentic_resolution_roi, \
            width_starting_point_authentic_resolution_roi, height_starting_point_authentic_resolution_roi, \
            width_tracking, height_tracking = data[DIMENSIONS_NAME]

        global_time_stamp = data[LOCAL_TIMESTAMP]
        if GLOBAL_TIMESTAMP_START_CONFIG in data:
            global_time_stamp += data[GLOBAL_TIMESTAMP_START_CONFIG]

        if self.min_viable_data_writer:
            # self.min_viable_data_writer.write_a_row(frame_number, data)
            self.min_viable_data_writer.write_a_row_for_tracklets(frame_number, data)

        self.timer += timeit.default_timer() - start_time
        return data

    def cleanup(self):
        super().cleanup()

        if self.min_viable_data_writer:
            self.min_viable_data_writer.finish_up()
