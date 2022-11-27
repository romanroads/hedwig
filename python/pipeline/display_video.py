import timeit
import cv2

from pipeline.pipeline import Pipeline
from constant_values import *


class DisplayVideo(Pipeline):
    def __init__(self, display_frame_name, output_width, output_height):
        self.display_frame_name = display_frame_name
        self.window_name = WINDOW_NAME
        self.stop_key = ESCAPE_KEY

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_EXPANDED)

        # Note we use output image frame aspect ratio to change the GUI window aspect ratio accordingly while
        # insisting the configured GUI height
        gui_height = HEIGHT_WINDOW
        gui_width = int(output_width * 1. / output_height * gui_height)
        cv2.resizeWindow(self.window_name, gui_width, gui_height)

        cv2.moveWindow(self.window_name, WINDOW_START_WIDTH, 10)

        super().__init__("DisplayVideo")

    def map(self, data):
        start_time = timeit.default_timer()
        image = data[self.display_frame_name]

        cv2.imshow(self.window_name, image)

        # Note Escape key for stop
        key = cv2.waitKey(1) & 0xFF
        if key == self.stop_key or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise StopIteration

        self.timer += timeit.default_timer() - start_time
        return data

    def cleanup(self):
        super().cleanup()
        cv2.destroyWindow(self.window_name)
