import os
import sys
from pathlib import Path
import logging

try:
    from drone_data_configuration import ConfigurationTagType, DroneDataConfiguration
except ImportError:
    pass

from constant_values import CONFIG_FILE_FOLDER_NAME


def parse_config_json(video_file, exp, run, trip, tag, json_map):

    if "ConfigString" not in json_map:
        logging.error("parse_config: config json invalid!!")
        return None

    config_dict = json_map["ConfigString"]

    drone_data_config = DroneDataConfiguration(config_dict["VideoPath"], config_dict["LabelPath"],
                                               exp, run, trip, tag, video_file)

    for k, v in config_dict.items():
        if k == "Height":
            drone_data_config.map_video_to_height[video_file] = float(v)
        elif k == "TextScaleFactor":
            drone_data_config.map_video_to_text_scale_factor[video_file] = float(v)
        elif k == "VehicleIDCircleRadius":
            drone_data_config.map_video_to_vehicle_id_radius[video_file] = float(v)
        elif k == "VehicleIDCircleOffSetX":
            drone_data_config.map_video_to_vehicle_id_offset_x[video_file] = float(v)
        elif k == "FrameIDFontSize":
            drone_data_config.map_video_to_font_scale_frame_id[video_file] = float(v)
        elif k == "Location":
            drone_data_config.map_video_to_location[video_file] = v
        elif k == "Date":
            drone_data_config.load_date(video_file, v)
        elif k == "Polygons":
            num_of_polygons, timestamp_start, timestamp_end = len(v), 0, -1
            if timestamp_end == -1:
                timestamp_end = sys.maxsize

            # initialize the polygon config part
            drone_data_config.add_polygon_config_to_video_config(video_file, num_of_polygons)

            # load each polygon
            list_for_polygon_content = []
            for ii in range(len(v)):
                m_dict = v[ii]
                poly = "Polygon", m_dict["LaneDirection"], m_dict["SlotDirection"], m_dict["LaneID"], m_dict["SlotID"],\
                       m_dict["PolygonBoundary"], m_dict["RoadID"]
                list_for_polygon_content.append(poly)

            drone_data_config.load_polygons(video_file, list_for_polygon_content,
                                            timestamp_start, timestamp_end)

        elif k == "CalibraionPoints":
            num_of_calib_points = len(v)
            # load each polygon
            list_for_calib_point_content = []
            for ii in range(len(v)):
                poly = "CalibrationPoint", "%s,%s" % (v[ii]["PixelCoord"], v[ii]["LatLonCoord"])
                list_for_calib_point_content.append(poly)

            drone_data_config.load_calibration_points(video_file, num_of_calib_points, list_for_calib_point_content)

    return drone_data_config


# Note: this method below needs to be removed
def parse_config_file(video_file, exp, run, trip, tag, calib_src):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    path = Path(dirname)
    path_to_config_file = os.path.join(path.parent, CONFIG_FILE_FOLDER_NAME)

    # Note: switch to cloud version of config files for production
    if calib_src is None:
        f = os.path.join(path_to_config_file, "exp%s_run%s_trip%s.txt" % (exp, run, trip))
    else:
        f = calib_src

    if not os.path.exists(f):
        return None

    logging.debug("parse_config: we are parsing configuration file %s:" % f)

    map_tag_name_to_value = {t.name: t for t in ConfigurationTagType}

    config_contents = []
    with open(f) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            content = line.strip()
            content = content.split(':')
            config_contents.append(content)
            line = fp.readline()
            cnt += 1

    if len(config_contents) < 2:
        logging.error("parse_config: config file invalid!!")
        return

    row = 0
    assert config_contents[row][0] == "VideoPath", "[ERROR] parse_config: VideoPath key not found in config file!"
    assert config_contents[row + 1][0] == "LabelPath", "[ERROR] parse_config: LabelPath key not found in config file!"

    drone_data_config = DroneDataConfiguration(config_contents[row][1], config_contents[row + 1][1],
                                               exp, run, trip, tag, video_file)

    row += 2

    # there could be multiple "Video" in one config file
    while row < len(config_contents) and config_contents[row][0] == "Video":

        row += 1

        while row < len(config_contents) and config_contents[row][0] != "Video":
            key = config_contents[row][0]
            value = config_contents[row][1]

            if key == "Exp":
                exp_value = value

                row += 1
                run_key = config_contents[row][0]
                run_value = config_contents[row][1]
                assert run_key == "Run", "Run number not given!"

                row += 1
                trip_key = config_contents[row][0]
                trip_value = config_contents[row][1]
                assert trip_key == "Trip", "Trip number not given!"

                row += 1
                tag_key = config_contents[row][0]
                tag_value = config_contents[row][1]

                assert tag_key == "Tag", "Tag not given!"
                assert tag_value in map_tag_name_to_value, "tag value is wrong!"
                tag_value = map_tag_name_to_value[tag_value]

                row += 1
                drone_data_config.load_exp_run_trip_tag(video_file, exp_value, run_value, trip_value,
                                                                     tag_value)
                logging.debug("%s %s %s %s %s %s %s %s" % (key, exp_value, run_key, run_value, trip_key, trip_value,
                                                   tag_key, tag_value))

            elif key == "Label":
                drone_data_config.map_video_to_label_files[video_file] =\
                    os.path.join(drone_data_config.label_path, value)

                row += 1

            elif key == "Height":
                drone_data_config.map_video_to_height[video_file] = float(value)
                row += 1

            elif key == "TextScaleFactor":
                drone_data_config.map_video_to_text_scale_factor[video_file] = float(value)
                row += 1

            elif key == "VehicleIDCircleRadius":
                drone_data_config.map_video_to_vehicle_id_radius[video_file] = float(value)
                row += 1

            elif key == "VehicleIDCircleOffSetX":
                drone_data_config.map_video_to_vehicle_id_offset_x[video_file] = float(value)
                row += 1

            elif key == "FrameIDFontSize":
                drone_data_config.map_video_to_font_scale_frame_id[video_file] = float(value)
                row += 1

            elif key == "Location":
                drone_data_config.map_video_to_location[video_file] = value
                row += 1

            elif key == "Date":
                drone_data_config.load_date(video_file, value)
                row += 1

            elif key == "LaneSlot":
                drone_data_config.is_slot_longer_than_lane = True
                row += 1

            elif key == "Lanes":
                num_of_lanes = int(value)
                num_of_lane_markers = num_of_lanes + 1
                drone_data_config.load_video_config(video_file, num_of_lanes)
                drone_data_config.load_lane_markers(video_file, config_contents[row + 1: row + 1 + num_of_lane_markers])

                row += num_of_lane_markers + 1

            elif key == "Slots":
                num_of_slots = int(value)
                num_of_slot_markers = num_of_slots + 1
                drone_data_config.add_slot_config_to_video_config(video_file, num_of_slots)
                drone_data_config.load_slot_markers(video_file, config_contents[row + 1: row + 1 + num_of_slot_markers])

                row += num_of_slot_markers + 1

            elif key == "Polygons":
                num_of_polygons, timestamp_start, timestamp_end = value.strip().split(',')
                num_of_polygons, timestamp_start, timestamp_end = int(num_of_polygons),\
                                                                  int(timestamp_start), int(timestamp_end)
                if timestamp_end == -1:
                    timestamp_end = sys.maxsize

                # initialize the polygon config part
                drone_data_config.add_polygon_config_to_video_config(video_file, num_of_polygons)

                # load each polygon
                drone_data_config.load_polygons(video_file, config_contents[row + 1: row + 1 + num_of_polygons],
                                                timestamp_start, timestamp_end)
                row += num_of_polygons + 1

            elif key == "Calibrations":
                calibration_key, num_of_calib_points = key, value
                assert calibration_key == "Calibrations", "Calibration key not found!!"
                num_of_calib_points = int(num_of_calib_points)
                drone_data_config.load_calibration_points(video_file, num_of_calib_points,
                                                          config_contents[row + 1: row + 1 + num_of_calib_points])

                row += num_of_calib_points + 1

    return drone_data_config


def parse_realtime_config(video_file, exp, run, trip, tag, list_static_tracklets):
    logging.debug("[INFO] parse_config: we are parsing configuration on-the-fly")
    drone_data_config = DroneDataConfiguration('Drone_Data', 'Drone_Data', exp, run, trip, tag, video_file)
    drone_data_config.load_date(video_file, "2020,2,21,15,0,0")

    drone_data_config.add_polygon_config_to_video_config(video_file, 0)
    drone_data_config.load_calibration_points(video_file, 2, [["CalibrationPoint", "2016,845,37.411417,-122.106536"],
                                                              ["CalibrationPoint", "2482,1112,37.411511,-122.106295"]])
    drone_data_config.map_video_to_location[video_file] = "Mountain View"
    drone_data_config.add_realtime_static_tracklets(list_static_tracklets)
    return drone_data_config


def main():
    parse_config_file("drone_data_config.txt")


if __name__ == "__main__":
    main()
