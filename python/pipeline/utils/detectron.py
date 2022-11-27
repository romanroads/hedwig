import os
import sys
import collections

import torch
import pickle
import logging

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from constant_values import *


def setup_cfg(model_path, dump_pickle_file, is_use_cpu_requested, user_id, license_id, version_id):
    """
    model_path is the folder pointing the model file, we use default model file name defined in constant_values.py
    :param model_path:
    :param is_use_cpu_requested:
    :return:
    """
    cfg = get_cfg()

    # Note: model_zoo.get_config_file will return a file path to yaml file which should point to conda area
    # under ~/miniconda3/envs/perception/lib/python3.6/site-packages/detectron2/model_zoo/configs/ which
    # means you should NOT mount or git clone source code of detectron2
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = (TRAINING_DATASET_NAME,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    ims_per_batch = 2
    base_learning_rate = 0.00025
    num_roi_per_image = 128
    num_classes = len(MAP_CATEGORY_ID_TO_NAME)
    num_key_points_for_single_class_detection = len(FEATURE_POINT_MAPS["Sedan"]["names"])

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_learning_rate
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = num_roi_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_key_points_for_single_class_detection
    cfg.MODEL.KEYPOINT_ON = True if num_key_points_for_single_class_detection > 0 else False

    cfg.VERSION = 2
    cfg.OUTPUT_DIR = model_path

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "%s" % RR_NET_MODEL_NAME)

    # Note if neither of user input or default exists, we start to download from cloud
    if not os.path.exists(cfg.MODEL.WEIGHTS):
        logging.warning("model: %s does NOT contain the DNN model file!" % model_path)
        sys.exit()

    confidence_threshold = min(THRESHOLD_DETECTION_PRECISION.values())
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.DATASETS.TEST = (TRAINING_DATASET_NAME,)

    if is_use_cpu_requested or not torch.cuda.is_available():
        logging.warning("we are using CPU for computing...")
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()

    if dump_pickle_file and ".pth" in cfg.MODEL.WEIGHTS:
        # Note try to save a pkl file of the model
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
        pickle_file_path = cfg.MODEL.WEIGHTS.replace(".pth", ".pkl")
        pickle_out = open(pickle_file_path, "wb")
        model_state_dict = model.state_dict()
        model_state_dict_numpy_array = collections.OrderedDict()
        for k, v in model_state_dict.items():

            # Note: these two guys does not belong to network weights
            if k == "pixel_mean" or k == "pixel_std":
                continue

            # Note: keypoints heads should also be included when converting to pb file for tensorflow
            # if "roi_heads.keypoint_head" in k:
            #     continue

            model_state_dict_numpy_array[k] = v.detach().to('cpu').numpy()

        dict_to_dump = {"model": model_state_dict_numpy_array, '__author__': 'ROMAN ROADS'}
        pickle.dump(dict_to_dump, pickle_out)
        pickle_out.close()

    return cfg
