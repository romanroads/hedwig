import timeit
import geopy.distance
import numpy as np

from stablization import Stablization

from parse_config import parse_config_json
from user_selected_roi import make_polygons
from pipeline.pipeline import Pipeline

from constant_values import *
from common.image_mapping import ImageMapping


class AutoCalibration(Pipeline):
    """
    what is the difference between calibration and configuration, config is part of calib.....
    config is for lane, slot measurement while calib is for lat, long measurement, etc.
    """
    def __init__(self, input_video_width, input_video_height, load_config, video_src, exp, run_number, trip, tag,
                 image_mapping, calibration_id, user_id, config_json):

        self.input_video_width = input_video_width
        self.input_video_height = input_video_height

        self.list_polygons_registration_zone = []
        self.list_polygons_unsubscription_zone = []
        self.list_occ_zone = []

        self.exp = exp
        self.run_number = run_number
        self.trip = trip
        self.tag = tag

        self.calibration_id = calibration_id
        self.user_id = user_id

        self.config_json = config_json

        if image_mapping:
            self.image_mapping = ImageMapping(40, 0, 0, -0.003, 0.000, 0.0003, self.input_video_width,
                                              self.input_video_height, z_ref=0.0005,
                                              pixel_size_at_projection_plane=0.000001)
        else:
            self.image_mapping = ImageMapping(0, 0, 0, 0, 0, 0, self.input_video_width,
                                              self.input_video_height)

        # Note: stablization module can still work on car traces even if config is not given for lane polygons
        self.stablization = Stablization(None, video_src)

        if load_config and self.config_json is not None:
            self.config = parse_config_json(video_src, self.exp, self.run_number, self.trip, self.tag, self.config_json)

            video_config = self.config.map_video_to_configs[video_src]

            for i in range(len(video_config.calibration_point_coordinates) - 1):
                self.calib_point_a = video_config.calibration_point_coordinates[i]
                self.calib_point_b = video_config.calibration_point_coordinates[i + 1]
        else:
            self.config = None
            self.calib_point_a = None
            self.calib_point_b = None

        super().__init__("AutoCalibration")

    def map(self, data):
        start_time = timeit.default_timer()
        frame_number = data[FRAME_NUMBER_NAME]
        video_name = data[VIDEO_NAME]

        if frame_number == 0:
            self.build_initial_calibration(data)

            if self.stablization is not None and STABLIZATION_ZONE_NAME in data and \
                    data[STABLIZATION_ZONE_NAME] is not None:
                list_sta_zones = data[STABLIZATION_ZONE_NAME]
                
                num_sta = sum([0 if len(sz) <= 0 else 1 for sz in list_sta_zones])
                if num_sta > 0:
                    width_authentic_resolution_roi, \
                    height_authentic_resolution_roi, \
                    width_starting_point_authentic_resolution_roi, \
                    height_starting_point_authentic_resolution_roi, width_tracking, height_tracking = \
                        data[DIMENSIONS_NAME]

                    ori_image_width, ori_image_height = data[ORI_FRAME_DIMEN_NAME]

                    self.stablization.set_stabalization_list(list_sta_zones)
                    self.stablization.set_center(width_starting_point_authentic_resolution_roi,
                                                 height_starting_point_authentic_resolution_roi,
                                                 width_authentic_resolution_roi,
                                                 height_authentic_resolution_roi,
                                                 ori_image_width, ori_image_height,
                                                 width_tracking,
                                                 height_tracking
                                                 )

                    list_polygons_stalization_zone = make_polygons(list_sta_zones,
                                                                   width_starting_point_authentic_resolution_roi,
                                                                   height_starting_point_authentic_resolution_roi,
                                                                   ori_image_width,
                                                                   ori_image_height,
                                                                   width_tracking,
                                                                   height_tracking,
                                                                   width_authentic_resolution_roi,
                                                                   height_authentic_resolution_roi)

                    self.stablization.set_stabalization_polygons(list_polygons_stalization_zone)
                else:
                    self.stablization = None

        data[CALIBRATION_ZONE_NAME] = [self.list_polygons_registration_zone, self.list_polygons_unsubscription_zone, self.list_occ_zone]

        if self.config is not None:
            data[CONFIG_NAME] = self.config
            data[CALIBRATION_POINT_A] = self.calib_point_a
            data[CALIBRATION_POINT_B] = self.calib_point_b

            calib_point_a = data[CALIBRATION_POINT_A]
            calib_point_b = data[CALIBRATION_POINT_B]
            x_pixel_a, y_pixel_a = calib_point_a.pixel
            x_pixel_b, y_pixel_b = calib_point_b.pixel
            latitude_a, longitude_a = calib_point_a.gps_location
            latitude_b, longitude_b = calib_point_b.gps_location
            distance_meter = geopy.distance.vincenty((latitude_a, longitude_a), (latitude_b, longitude_b)).km * 1000.
            distance_pixels = np.sqrt((x_pixel_a - x_pixel_b) ** 2 + (y_pixel_a - y_pixel_b) ** 2)

            data[CALIBRATION_RESOLUTION] = distance_meter / distance_pixels

            data[GLOBAL_TIMESTAMP_START_CONFIG] = self.config.map_video_to_date[video_name]
            data[LOCATION_NAME_CONFIG] = self.config.map_video_to_location[video_name]

        data[CALIBRATION_IMAGE_MAPPING_NAME] = self.image_mapping

        if self.stablization is not None:
            data[STABLIZATION_SYSTEM_NAME] = self.stablization

        self.timer += timeit.default_timer() - start_time
        return data

    def build_initial_calibration(self, data):
        width_authentic_resolution_roi, height_authentic_resolution_roi, width_starting_point_authentic_resolution_roi,\
            height_starting_point_authentic_resolution_roi, width_tracking, height_tracking = data[DIMENSIONS_NAME]

        list_frac_pixel_cords_roi_registration_zone, list_frac_pixel_cords_roi_unsubscription_zone, list_occlusion =\
            data[CALIBRATION_ZONE_POINTS_NAME]

        self.list_polygons_registration_zone = make_polygons(list_frac_pixel_cords_roi_registration_zone,
                                                             width_starting_point_authentic_resolution_roi,
                                                             height_starting_point_authentic_resolution_roi,
                                                             self.input_video_width,
                                                             self.input_video_height,
                                                             width_tracking,
                                                             height_tracking,
                                                             width_authentic_resolution_roi,
                                                             height_authentic_resolution_roi)

        self.list_polygons_unsubscription_zone = make_polygons(
            list_frac_pixel_cords_roi_unsubscription_zone,
            width_starting_point_authentic_resolution_roi,
            height_starting_point_authentic_resolution_roi,
            self.input_video_width,
            self.input_video_height,
            width_tracking,
            height_tracking,
            width_authentic_resolution_roi,
            height_authentic_resolution_roi)

        self.list_occ_zone = make_polygons(
            list_occlusion,
            width_starting_point_authentic_resolution_roi,
            height_starting_point_authentic_resolution_roi,
            self.input_video_width,
            self.input_video_height,
            width_tracking,
            height_tracking,
            width_authentic_resolution_roi,
            height_authentic_resolution_roi)


if __name__ == "__main__":
    auto_calib = AutoCalibration(1, 1, True, "", 0, 0, 0, "User", None, 2, "us", 6)
