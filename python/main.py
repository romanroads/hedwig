import os
import io
import optparse
from tqdm import tqdm
import multiprocessing as mp
import logging

from pipeline.capture_video import CaptureVideo
from pipeline.prepare_images import PrepareImages
from pipeline.process_predictions import ProcessPredictions
from pipeline.auto_calibration import AutoCalibration
from pipeline.async_predict import AsyncPredict
from pipeline.track_agents_individualized import TrackAgentsIndividualized
from pipeline.annotate_video import AnnotateVideo
from pipeline.display_video import DisplayVideo
from pipeline.save_video import SaveVideo
from pipeline.upload_to_cloud import UploadToCloud

from constant_values import *


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


def main():
    parser = optparse.OptionParser()
    parser.add_option('--user_id', action="store", default=0)
    parser.add_option('--license_id', action="store", default="0")
    parser.add_option('--model_path', action="store", default="./",
                      help="the folder that contains the model file")
    parser.add_option('--version_id', action="store", default=0)
    parser.add_option('-e', '--exp', action="store", default=21)
    parser.add_option('-r', '--run', action="store", default=37)
    parser.add_option('-t', '--trip', action="store", default=0)
    parser.add_option('-f', '--video_file', action="store", default="test_data.mp4")
    parser.add_option('--video_path', action="store", default="./")
    parser.add_option('--tag', dest="tag", action="store", default="Sample",
                      help="tags for data: Production, Sample, User")
    parser.add_option('-v', '--vehicle_id_start', action='store', default=0)
    parser.add_option('--max_frame', action="store", default=-1)
    parser.add_option('--min_frame', action="store", default=0)
    parser.add_option('-x', '--width_resize', action='store', default=-1)
    parser.add_option('-y', '--height_resize', action='store', default=-1)
    parser.add_option('--tracking_x', action='store', default=-1)
    parser.add_option('--tracking_y', action='store', default=-1)
    parser.add_option('-a', '--width_output', action='store', default=-1)
    parser.add_option('-b', '--height_output', action='store', default=-1)
    parser.add_option('-s', '--mask', action='store_true', default=False)
    parser.add_option('--box', action='store_true', default=False)
    parser.add_option('--polygon', action='store_true', default=False)
    parser.add_option('--tracklet_traj', action='store_true', default=False)
    parser.add_option('--tracklet_feature', action='store_true', default=False, help="optical flow feature points")
    parser.add_option('--road_markers', action='store_true', default=False)
    parser.add_option('--reg_unsub_zone', action='store_true', default=False)
    parser.add_option('--lane_id_slot_id_measurements', action='store_true', default=False)
    parser.add_option('--car_heading', action='store_true', default=False)
    parser.add_option('--feature_points', action='store_true', default=False)
    parser.add_option('--resolution', action='store_true', default=False)
    parser.add_option('--stablization_zone', action='store_true', default="this plots stablization zone")
    parser.add_option('--stablization_correction', action='store_true', default="this plots stablization corrections")
    parser.add_option('--stablization_zone_manual_input', action="store", default="",
                      help="the list of list of coordinates, eg.: 0.1,0.2,0.3,0.4;0.11,0.22,0.33,0.44")
    parser.add_option('--roi_manual_input', action="store", default="",
                      help="the list of list of coordinates, eg.: 0.1,0.2,0.3,0.4")
    parser.add_option('--reg_zone_manual_input', action="store", default="",
                      help="the list of list of coordinates, eg.: 0.1,0.2,0.3,0.4;0.11,0.22,0.33,0.44")
    parser.add_option('--unsub_zone_manual_input', action="store", default="",
                      help="the list of list of coordinates, eg.: 0.1,0.2,0.3,0.4;0.11,0.22,0.33,0.44")
    parser.add_option('--auto', action='store_true', default=False, help="automatically start detection + tracking,"
                                                                         "for batching processing jobs, no display")
    parser.add_option('--load_config', action="store_true",
                      default=False, help="whether or not to load config (calib) files, lane id will be available")
    parser.add_option('--calibration_id', action="store", default=-1)
    parser.add_option('--auto_config', action="store_true",
                      default=False, help="whether or not to generate config on-the-fly")
    parser.add_option('--avs', action="store", default="")
    parser.add_option('--commit', action="store_true", default=False)

    parser.add_option('--commit_user_req_data', action="store_true", default=False)
    parser.add_option('--commit_min_viable_data', action="store_true", default=False)
    parser.add_option('--is_velo_sample_slow_down', action="store_true", default=False)
    parser.add_option('--is_acc_sample_slow_down', action="store_true", default=False)
    parser.add_option('--velo_sample_rate', action='store', default=1.0)
    parser.add_option('--acc_sample_rate', action='store', default=1.0)

    parser.add_option('--is_fit_req_to_speed', action="store_true", default=False)
    parser.add_option('--speed_poly_fit_order', action='store', default=3)

    parser.add_option('--is_periodically_upload_user_data', action="store_true", default=False)
    parser.add_option('--upload_interval_user_data', action='store', default=10)
    parser.add_option('--num_digits_for_print', action='store', default=4)
    parser.add_option('--last_commit_num_chunks', action='store', default=1)

    parser.add_option('--is_initial_lane_polygon_adjusted_req', action="store_true", default=False)
    parser.add_option('--initial_rot_angle', action='store', default=0.)
    parser.add_option('--initial_x', action='store', default=0.)
    parser.add_option('--initial_y', action='store', default=0.)

    parser.add_option('--plot_vector_image', action="store_true", default=False)

    parser.add_option('--is_to_correct_vehicles', action="store_true", default=False)

    parser.add_option('--csv', action="store_true", default=False)
    parser.add_option('--download', action="store_true", default=False)
    parser.add_option('--cloud', action="store", default="US")
    parser.add_option('--calibrate', action='store_true', default=False)
    parser.add_option('--bottom_right', action='store_true', default=False)
    parser.add_option('--logo_pos', action='store', default="bottom_right")
    parser.add_option("--gpus", type=int, default=1,
                    help="number of GPUs (default: 1)")
    parser.add_option("--cpus", type=int, default=0,
                    help="number of CPUs (default: 0)")
    parser.add_option("--queue-size", type=int, default=3,
                    help="queue size per process (default: 3)")
    parser.add_option('--image_mapping', action="store_true", default=False)
    parser.add_option('--plot_warped_image', action="store_true", default=False)
    parser.add_option('--feature_detector', action="store", default="gftt",
                      help="types of feature detector: orb, gftt, none (using DNN feature points) ...")
    parser.add_option('--logging', action="store", default="info", help="debug, info, warning, error")
    parser.add_option('--dump_pickle_file', action="store_true",
                      default=False, help="whether or not to dump model file to pickle file")

    parser.add_option('--thresh_det', action='store',
                      default="Sedan:0.6,Truck:0.6,Motorcycle:0.7,Bicycle:0.7,Pedestrian:0.8,RoadFeatures:0.8")

    parser.add_option('--center_dot_size', action='store', default=16)
    parser.add_option('--traj_width', action='store', default=4)
    parser.add_option('--agent_id_text_size', action='store', default=1)
    parser.add_option('--agent_id_font_size', action='store', default=1.0)
    parser.add_option('--agent_id_offset_w', action='store', default=-40)
    parser.add_option('--agent_id_offset_h', action='store', default=12)

    parser.add_option('--initial_time_window_for_reg', action='store', default=0.5)

    parser.add_option('--constrain_x_y_into_lane_polygon', action="store_true", default=False)

    parser.add_option('--is_to_removed_bad_quality_tracklets', action="store_true", default=False)
    parser.add_option('--bad_quality_tracklets_dist', action='store', default=10)

    parser.add_option('--detection_frame_gap', action='store', default=4)

    options, args = parser.parse_args()

    if options.logging.lower() == "info":
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

    logger = logging.getLogger()
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    video_src = os.path.join(options.video_path, options.video_file)

    if not os.path.exists(video_src):
        logging.warning("main: video file %s can not be found" % video_src)
        return

    logging.info("main: we would like to work on data %s" % video_src)

    # Note, boilerplate for cloud-based config
    config_json = None

    # Note: define the various sequential stages of pipeline, and use pipes to connect them
    capture_video = CaptureVideo(video_src, options.min_frame, options.max_frame)

    prepare_images = PrepareImages(video_src, capture_video.input_video_width, capture_video.input_video_height,
                                   options.width_output, options.height_output, options.width_resize,
                                   options.height_resize, options.tracking_x, options.tracking_y, options.auto,
                                   options.calibrate, config_json, options.stablization_zone_manual_input,
                                   options.plot_vector_image, options.roi_manual_input, options.reg_zone_manual_input,
                                   options.unsub_zone_manual_input)

    mp.set_start_method("spawn", force=True)

    predict = AsyncPredict(options.model_path, options.dump_pickle_file, options.user_id, options.license_id,
                           options.version_id, int(options.detection_frame_gap),
                           num_gpus=options.gpus, num_cpus=options.cpus,
                           queue_size=options.queue_size, ordered=True)

    processed_predictions = ProcessPredictions(prepare_images.width_tracking,
                                               prepare_images.height_tracking,
                                               prepare_images.width_detection,
                                               prepare_images.height_detection,
                                   prepare_images.scale_width_from_det_to_track,
                                   prepare_images.scale_height_from_det_to_track,
                                   prepare_images.scale_width_from_det_to_output,
                                   prepare_images.scale_height_from_det_to_output,
                                               options.thresh_det)

    auto_calibration = AutoCalibration(capture_video.input_video_width, capture_video.input_video_height,
                                       options.load_config, video_src,
                                       options.exp, options.run, options.trip, options.tag, options.image_mapping,
                                       options.calibration_id, options.user_id, config_json)

    track_agents_individualized = TrackAgentsIndividualized(
        processed_predictions.num_row, processed_predictions.num_col, processed_predictions.num_grid_cells, capture_video.fps,
        int(options.vehicle_id_start), options.avs, options.feature_detector, float(options.initial_rot_angle), float(options.initial_x), float(options.initial_y),
        options.is_initial_lane_polygon_adjusted_req, options.is_to_correct_vehicles, float(options.initial_time_window_for_reg))

    annotate_video = AnnotateVideo(prepare_images.scale_width_from_tracking_to_output,
                                   prepare_images.scale_height_from_tracking_to_output,
                                   options.polygon, options.box, options.mask, options.road_markers,
                                   options.lane_id_slot_id_measurements,
                                   options.reg_unsub_zone, options.tracklet_traj,
                                   options.tracklet_feature,
                                   options.stablization_zone,
                                   options.stablization_correction, prepare_images.width_output,
                                   prepare_images.height_output, not options.bottom_right, options.logo_pos,
                                   options.plot_warped_image, options.car_heading, options.feature_points,
                                   options.resolution, options.center_dot_size, options.traj_width,
                                   options.agent_id_text_size, options.agent_id_font_size, options.agent_id_offset_w,
                                   options.agent_id_offset_h)

    display_video = DisplayVideo(OUTPUT_IMAGE_NAME, int(options.width_output), int(options.height_output))

    save_video = SaveVideo(video_src, options.exp, options.run, options.trip, options.tag, prepare_images.width_output,
                    prepare_images.height_output, user_id=options.user_id, license_id=options.license_id)

    upload_to_cloud = UploadToCloud(options.exp, options.run, options.trip, options.tag,
                                    capture_video.fps, options.commit, options.cloud, options.auto, options.csv,
                                    options.commit_user_req_data,
                                    options.commit_min_viable_data,
                                    options.is_velo_sample_slow_down,
                                    float(options.velo_sample_rate),
                                    options.is_acc_sample_slow_down,
                                    float(options.acc_sample_rate),
                                    options.is_fit_req_to_speed,
                                    int(options.speed_poly_fit_order),
                                    options.is_periodically_upload_user_data,
                                    int(options.upload_interval_user_data),
                                    int(options.num_digits_for_print),
                                    int(options.last_commit_num_chunks),
                                    options.constrain_x_y_into_lane_polygon,
                                    options.is_to_removed_bad_quality_tracklets,
                                    float(options.bad_quality_tracklets_dist),
                                    user_id=options.user_id,
                                    license_id=options.license_id)

    stages = [capture_video, prepare_images, predict, processed_predictions, auto_calibration,
              track_agents_individualized, annotate_video, display_video, save_video, upload_to_cloud]

    pipeline = None
    for stage in stages:
        if pipeline is None:
            pipeline = stage
        else:
            pipeline |= stage

    try:
        tqdm_pipeline = tqdm(pipeline, total=capture_video.frame_count if capture_video.frame_count > 0 else None,
                             disable=False, file=tqdm_out, mininterval=3)

        for _ in tqdm_pipeline:
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        for stage in stages:
            stage.cleanup()


if __name__ == "__main__":
    main()
