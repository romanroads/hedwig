import timeit
from collections import deque

from pipeline.pipeline import Pipeline
from pipeline.libs.async_predictor import AsyncPredictor
from pipeline.utils import detectron
from constant_values import *


class AsyncPredict(Pipeline):
    """perform prediction asynchronously in separate processes"""

    def __init__(self, model_path, dump_pickle_file, user_id, license_id, version_id, detection_frame_gap,
                 num_gpus=1, num_cpus=1, queue_size=3, ordered=True):

        cfg = detectron.setup_cfg(model_path, dump_pickle_file, False if num_gpus > 0 else True, user_id, license_id,
                                  version_id)

        self.dump_pickle_file = dump_pickle_file
        self.predictor = AsyncPredictor(cfg,
                                        num_gpus=num_gpus,
                                        num_cpus=num_cpus,
                                        queue_size=queue_size,
                                        ordered=ordered)
        self.ordered = ordered
        self.buffer_size = self.predictor.num_procs * queue_size
        self.frame_counter = 0
        self.detection_frame_gap = detection_frame_gap

        super().__init__("AsyncPredict")

    def generator(self):
        if self.ordered:
            return self.serial_generator()
        else:
            return self.parallel_generator()

    def serial_generator(self):
        buffer = deque()
        buffer_to_process = deque()
        stop = False
        buffer_cnt = 0
        while self.has_next() and not stop:
            try:
                data = next(self.source)

                self.frame_counter = data[FRAME_NUMBER_NAME]
                buffer.append(data)

                if self.frame_counter % self.detection_frame_gap == 0:
                    start_time = timeit.default_timer()
                    self.predictor.put(data[DETECTION_IMAGE_NAME])
                    self.timer += timeit.default_timer() - start_time

                    buffer_to_process.append(True)
                else:
                    buffer_to_process.append(False)

                buffer_cnt += 1

            except StopIteration:
                stop = True

            if buffer_cnt >= self.buffer_size:

                is_to_process = buffer_to_process.popleft()
                if is_to_process:
                    start_time = timeit.default_timer()
                    predictions = self.predictor.get()
                    self.timer += timeit.default_timer() - start_time
                else:
                    predictions = None

                data = buffer.popleft()
                data[PREDICTIONS_NAME] = predictions

                if self.filter(data):
                    yield self.map(data)

        while len(buffer):
            is_to_process = buffer_to_process.popleft()
            if is_to_process:
                start_time = timeit.default_timer()
                predictions = self.predictor.get()
                self.timer += timeit.default_timer() - start_time
            else:
                predictions = None

            data = buffer.popleft()
            data[PREDICTIONS_NAME] = predictions

            if self.filter(data):
                yield self.map(data)

    def parallel_generator(self):
        buffer = {}
        stop = False
        buffer_cnt = 0
        while self.has_next() and not stop:
            try:
                data = next(self.source)
                buffer[data["image_id"]] = data
                self.predictor.put((data["image_id"], data[DETECTION_IMAGE_NAME]))
                buffer_cnt += 1
            except StopIteration:
                stop = True

            if buffer_cnt >= self.buffer_size:
                image_id, predictions = self.predictor.get()
                data = buffer[image_id]
                data["predictions"] = predictions
                del buffer[image_id]

                if self.filter(data):
                    yield self.map(data)

        while len(buffer.keys()):
            image_id, predictions = self.predictor.get()
            data = buffer[image_id]
            data["predictions"] = predictions
            del buffer[image_id]

            if self.filter(data):
                yield self.map(data)

    def cleanup(self):
        super().cleanup()
        self.predictor.shutdown()
