import timeit
import cv2
import os
from pathlib import Path
import logging

from pipeline.pipeline import Pipeline
from utils import get_video_file_artifact_file_name
from constant_values import *


class SaveVideo(Pipeline):
    def __init__(self, video_file, exp, run_number, trip, tag, width_output, height_output, user_id=0, license_id="na"):

        _, self.video_file_name = get_video_file_artifact_file_name(video_file, "_processed.mp4")

        self.exp = exp
        self.run_number = run_number
        self.trip = trip
        self.tag = tag
        self.user_id = user_id
        self.license_id = license_id
        self.output_video_width = width_output
        self.output_video_height = height_output

        self.outvideo = None

        self.path_to_saved_video = None
        self.key_of_video_file_for_cloud = None

        self.setup_outvideo()
        super().__init__("SaveVideo")

    def setup_outvideo(self):
        dirname, _ = os.path.split(os.path.abspath(__file__))

        path_to_output = os.path.join(Path(dirname).parent.parent, PROCESSED_VIDEO_FOLDER_NAME)
        if os.path.exists(path_to_output) is False:
            os.mkdir(path_to_output)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        file_name = os.path.join(path_to_output, self.video_file_name)

        self.outvideo = cv2.VideoWriter(file_name, fourcc, FPS_OUTPUT, (self.output_video_width,
                                                                       self.output_video_height))

        self.path_to_saved_video = file_name
        self.key_of_video_file_for_cloud = self.video_file_name

        logging.debug("save_video: output video %s dimension chosen to be (W %s x H %s):" %
              (file_name, self.output_video_width, self.output_video_height))

        if self.outvideo.isOpened():
            logging.debug("save_video: Create output video %s success!" % file_name)
        else:
            logging.debug("save_video: Create output video %s failed!" % file_name)

    def map(self, data):
        start_time = timeit.default_timer()
        output_image = data[OUTPUT_IMAGE_NAME]

        if self.outvideo is None:
            return

        self.outvideo.write(output_image)
        data[SAVED_VIDEO_FILE_PATH] = self.path_to_saved_video
        data[SAVED_VIDEO_FILE_CLOUD_KEY] = self.key_of_video_file_for_cloud

        self.timer += timeit.default_timer() - start_time
        return data

    def cleanup(self):
        super().cleanup()
        if self.outvideo:
            self.outvideo.release()
