import timeit
import cv2 as cv

from pipeline.pipeline import Pipeline
from pipeline.libs.file_video_capture import FileVideoCapture
from pipeline.libs.webcam_video_capture import WebcamVideoCapture

from constant_values import RAW_IMAGE_NAME, FRAME_NUMBER_NAME, FRAME_COUNT_NAME, IMAGE_ID_NAME, VIDEO_NAME,\
    LOCAL_TIMESTAMP, ORI_FRAME_COUNT_NAME, ORI_FRAME_DIMEN_NAME, FPS_NAME


class CaptureVideo(Pipeline):
    def __init__(self, src, min_frame, max_frame):
        # Note min_frame default is 0 while max_frame default is -1 standing for infinity
        self.min_frame = int(min_frame)
        self.max_frame = int(max_frame)

        if isinstance(src, int):
            self.cap = WebcamVideoCapture(src).start()
            self.frame_count = -1
            self.ori_frame_count = self.frame_count
        else:
            self.cap = FileVideoCapture(src, self.min_frame).start()
            self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            self.ori_frame_count = self.frame_count

            if 0 <= self.max_frame < self.frame_count:
                self.frame_count = self.max_frame + 1

            if 0 <= self.min_frame < self.frame_count:
                self.frame_count = self.frame_count - self.min_frame

        self.fps = float(self.cap.get(cv.CAP_PROP_FPS))
        self.period_in_ms = 1. / self.fps * 1000.
        self.input_video_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.input_video_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.input_video_width, self.input_video_height)

        self.frame_num = 0
        self.src = src

        self.starting_local_timestamp = None

        super().__init__("CaptureVideo")

    def generator(self):
        while self.cap.running() and self.frame_num < self.frame_count:

            start_time = timeit.default_timer()
            image = self.cap.read()
            self.timer += timeit.default_timer() - start_time
            local_time_stamp = int(self.period_in_ms * self.frame_num)

            data = {
                FRAME_NUMBER_NAME: self.frame_num,
                FRAME_COUNT_NAME: self.frame_count,
                ORI_FRAME_COUNT_NAME: self.ori_frame_count,
                IMAGE_ID_NAME: f"{self.frame_num:06d}",
                RAW_IMAGE_NAME: image,
                VIDEO_NAME: self.src,
                LOCAL_TIMESTAMP: local_time_stamp,
                ORI_FRAME_DIMEN_NAME: self.frame_size,
                FPS_NAME: self.fps
            }

            self.frame_num += 1
            if self.filter(data):
                yield self.map(data)

    def cleanup(self):
        super().cleanup()
        self.cap.stop()
