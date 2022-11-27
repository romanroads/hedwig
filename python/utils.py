import os
import glob
import numpy as np
import json
import cv2
import itertools
import math
import logging
from sqlalchemy import create_engine

try:
    from shapely import geometry
    from shapely.ops import cascaded_union, polygonize
    from scipy.spatial import Delaunay
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
    from detectron2.structures import BoxMode
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
except:
    pass

from constant_values import *
from user_selected_roi import get_roi
from user_selected_mask import get_rect_roi, get_masked_image_with_roi


def get_car_dicts(img_dir="./", json_file="via_region_data.json"):
    json_file = os.path.join(img_dir, json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['_via_img_metadata'].values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for anno in annos:
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(img_dir="./", json_file="via_region_data.json"):
    car_dicts = get_car_dicts(img_dir, json_file)
    DatasetCatalog.register("rr_car", lambda: car_dicts)
    # one class for now
    MetadataCatalog.get("rr_car").set(thing_classes=["rr_car"])
    car_metadata = MetadataCatalog.get("rr_car")
    return car_metadata, car_dicts


def get_poly(line):
    """
    this is the main work horse to read and parse label, annotation text files
    this function is used by training, by checking annotation
    :param line:
    :return:
    """
    points = line.split()
    agent_id = int(points[0])
    num_points = int(points[1])
    num_key_points = 0 if len(points) <= num_points * 2 + 3 else int(points[num_points * 2 + 3])
    if num_key_points <= 0:
        assert len(points) == num_points * 2 + 3,\
            "[ERROR] utils: data entry %s seems to be having corrupted structures" % line
    else:
        assert len(points) == num_points * 2 + 3 + num_key_points * 3 + 1, \
            "[ERROR] utils: data entry %s seems to be having corrupted structures" % line

    poly = []
    feature_points = []
    max_x, max_y = 0, 0
    min_x, min_y = np.inf, np.inf

    # Note a polygon must be defined with at least 3 points
    if num_points < 3:
        print("[WARNING] mistakes in the annotation, line %s num points %s for polygon" % (line, num_points))
        return None

    for i in range(num_points):
        x = float(points[2 * i + 2])
        y = float(points[2 * i + 3])
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        poly.append((x, y))

    for i in range(num_key_points):
        x = float(points[3 * i + num_points * 2 + 3 + 1])
        y = float(points[3 * i + num_points * 2 + 3 + 2])
        visible = float(points[3 * i + num_points * 2 + 3 + 3])
        feature_points.append((x, y, visible))

    # Note: when the text annotation file contains labels for feature points, but the constants defines
    # empty or zero feature points, that means we do not wanna train using feature point data, reset it to null
    num_key_points_for_single_class_detection = len(FEATURE_POINT_MAPS["Sedan"]["names"])
    if num_key_points_for_single_class_detection <= 0:
        feature_points = []

    category_name = points[num_points * 2 + 3 - 1]
    return agent_id, poly, [min_x, min_y, max_x, max_y], category_name, feature_points


def get_poly_shapely(line):
    points = line.split()
    agent_id = int(points[0])
    list_points = []
    num_points = int(points[1])
    for i in range(num_points):
        x = float(points[2 * i + 2])
        y = float(points[2 * i + 3])
        list_points.append((x, y))

    polygon = geometry.Polygon(list_points)
    box = polygon.minimum_rotated_rectangle

    return agent_id, polygon, box


def get_custom_data_dicts(img_dir, label_dir, custom_data_dir, synthetic_batch_for_training):
    """
    set synthetic_batch_for_training to empty string if no auto selection of data is needed
    :param img_dir:
    :param label_dir:
    :param custom_data_dir: the folder names under this folder should be either Exp*/Run*/Images(Texts)
    or Synthetic_Batch_x
    :param synthetic_batch_for_training: '1,2' or ''
    :return:
    """
    img_dirs = []
    label_dirs = []
    if custom_data_dir is not None:
        if len(synthetic_batch_for_training) > 0:
            logging.info("batch IDs are provided: %s, usually this means the data are synthetic data" %
                  synthetic_batch_for_training)
            folder_indicies = synthetic_batch_for_training.split(',')

            list_folders = glob.glob(os.path.join(custom_data_dir, "*/"))
            for folder in list_folders:
                base_folder_name = folder.split('/')[-2]
                batch_index = base_folder_name.split('_')[2]
                batch_keyword = base_folder_name.split('_')[1]
                assert batch_keyword == "Batch", "[ERROR] utils: the folder naming structure under %s wrong" %\
                                                 custom_data_dir
                if batch_index not in folder_indicies:
                    continue

                img_dirs.append(os.path.join(folder, "Images"))
                label_dirs.append(os.path.join(folder, "Texts"))
        else:
            logging.info("no batch ID provided, we will search for all text files under structure Exp*/Run*/ under %s" %
                  custom_data_dir)
            list_folders = glob.glob(os.path.join(custom_data_dir, "Exp*/Run*"))
            for folder in list_folders:
                img_dirs.append(os.path.join(folder, "Images"))
                label_dirs.append(os.path.join(folder, "Texts"))

    else:
        img_dirs.append(img_dir)
        label_dirs.append(label_dir)

    dataset_dicts = []
    classes = set()
    class_id_to_name = {}
    class_id_to_counts = {}

    for i in range(len(img_dirs)):
        img_dir = img_dirs[i]
        label_dir = label_dirs[i]

        # Note label directory can be not existing..... which means no annotation has been provided yet
        if not os.path.exists(label_dir):
            continue

        assert (os.path.isdir(img_dir)), "[ERROR] utils: image directory {} does not exist!".format(img_dir)

        # Note we rely on the text files for searching for annotations, since some of the images files may not
        # get labeled thus have no corresponding text files
        text_files = glob.glob(os.path.join(label_dir, "*.txt"))

        # Note there might be ROI files associated with each text files, let's remove them
        text_files_filtered = []
        for _text_file in text_files:
            if "_ROI.txt" not in _text_file and "_backup.txt" not in _text_file:
                text_files_filtered.append(_text_file)

        text_files = text_files_filtered

        for idx, text_file in enumerate(text_files):
            image_file = os.path.join(img_dir, os.path.basename(text_file).replace(".txt", ".jpg"))
            roi_file = text_file.replace(".txt", "_ROI.txt")

            if os.path.exists(roi_file):
                logging.info("found a roi file %s" % roi_file)

                image_file_with_roi_applied = image_file.replace(".jpg", "_ROI.jpg")
                if not os.path.exists(image_file_with_roi_applied):

                    frame = cv2.imread(image_file)
                    list_frac_pixel_cords_roi, list_frac_pixel_cords_roi_registration_zone, \
                    list_frac_pixel_cords_roi_unsubscription_zone, \
                    list_frac_pixel_cords_roi_occlusion_zone, \
                    list_frac_pixel_cords_roi_stablization_zone, is_everything_not_selected = \
                        get_roi(frame, os.path.dirname(roi_file), os.path.basename(roi_file), auto_processing=True)

                    rect_roi = get_rect_roi(frame, list_frac_pixel_cords_roi)

                    frame = get_masked_image_with_roi(frame, list_frac_pixel_cords_roi, rect_roi)
                    cv2.imwrite(image_file_with_roi_applied, frame)

                image_file = image_file_with_roi_applied

            record = {}
            height, width = cv2.imread(image_file).shape[:2]
            record["file_name"] = image_file
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            with open(text_file) as f:
                line = f.readline().strip()
                while line:

                    parsed_annotation = get_poly(line)
                    if parsed_annotation is None:
                        print("[WARNING] found a mistake in %s, skipping this line now...." % text_file)
                        line = f.readline().strip()
                        continue

                    agent_id, poly, box, category_name, key_points = parsed_annotation

                    classes.add(category_name)

                    poly = list(itertools.chain.from_iterable(poly))

                    assert category_name in MAP_CATEGORY_NAME_SEMANTIC_CLASS, "utils: seems data entry has undefined" \
                                                                              "category %s" % category_name

                    category_id = MAP_CATEGORY_NAME_SEMANTIC_CLASS[category_name]

                    assert category_id in MAP_CATEGORY_ID_TO_NAME, "utils: the category id %s not in our system" % \
                                                                   category_id

                    category_name_in_perception_system = MAP_CATEGORY_ID_TO_NAME[category_id]
                    class_id_to_name[category_id] = category_name_in_perception_system

                    if category_id not in class_id_to_counts:
                        class_id_to_counts[category_id] = 0
                    class_id_to_counts[category_id] += 1

                    if len(key_points) > 0:
                        obj = {
                            "bbox": box,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": category_id,
                            "iscrowd": 0,
                            "keypoints": key_points
                        }
                    else:
                        obj = {
                            "bbox": box,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": category_id,
                            "iscrowd": 0
                        }

                    objs.append(obj)
                    line = f.readline().strip()

            record["annotations"] = objs
            dataset_dicts.append(record)

    logging.info("utils: read %s semantic classes: %s" % (len(classes), classes))
    class_ids = list(class_id_to_name.keys())
    class_ids.sort()
    class_names = [class_id_to_name[class_id] for class_id in class_ids]
    class_counts = [class_id_to_counts[class_id] for class_id in class_ids]
    logging.info("utils: after sorting as double check, and convert from semantic class to classes in perceptionDNN, "
          "we read %s classes: %s ids %s counts %s" % (len(class_names), class_names, class_ids, class_counts))

    return dataset_dicts, class_ids, class_names


def register_image_and_label_dataset(img_dir, txt_dir, custom_data_dir, synthetic_batch_for_training):
    """
    set synthetic_batch_for_training to empty string if you do not intent to automate the data selection
    :param img_dir: can be None
    :param txt_dir: can be None
    :param custom_data_dir: the folder to locate the training images and label text files
    :param synthetic_batch_for_training: '1,2' or ''
    :return:
    """
    car_dicts, class_ids, class_names = get_custom_data_dicts(img_dir, txt_dir, custom_data_dir,
                                                              synthetic_batch_for_training)

    DatasetCatalog.register(TRAINING_DATASET_NAME, lambda: car_dicts)
    MetadataCatalog.get(TRAINING_DATASET_NAME).set(thing_classes=class_names)

    MetadataCatalog.get(TRAINING_DATASET_NAME).set(keypoint_names=FEATURE_POINT_MAPS['Sedan']['names'])
    MetadataCatalog.get(TRAINING_DATASET_NAME).set(keypoint_flip_map=FEATURE_POINT_MAPS['Sedan']['flip_map'])
    MetadataCatalog.get(TRAINING_DATASET_NAME).set(keypoint_connection_rules=
                                                   FEATURE_POINT_MAPS['Sedan']['connection_rules'])

    car_metadata = MetadataCatalog.get(TRAINING_DATASET_NAME)
    return car_metadata, car_dicts


class Detector:
    def __init__(self, model_path, thresh_classification, is_gpu=True):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
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

        if not is_gpu:
            cfg.MODEL.DEVICE = 'cpu'

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_rr_net.pth")

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_classification  # set the testing threshold for this model
        cfg.DATASETS.TEST = (TRAINING_DATASET_NAME,)

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.print()

    def predict(self, img):
        return self.predictor(img)

    def print(self):
        logging.info("utils: RR Net loaded model from: %s" % self.cfg.OUTPUT_DIR)


def get_scaled_mask(original_mask, desired_width, desired_height):
    """
    extrapolate or interpolate semantic maps
    """
    seg_map = SegmentationMapsOnImage(original_mask, shape=original_mask.shape)
    seg_map = seg_map.resize((desired_height, desired_width))
    scaled_mask = seg_map.get_arr()
    return scaled_mask


def is_in_box(x, y, box_center, bbox_width, bbox_height):
    b_x, b_y = box_center
    assert bbox_width > 0 and bbox_height > 0, "wrong box dimension given"
    # return b_x <= x <= b_x + bbox_width and b_y <= y <= b_y + bbox_height
    return b_x - bbox_width * 0.5 <= x <= b_x + bbox_width * 0.5 and b_y - \
           bbox_height * 0.5 <= y <= b_y + bbox_height * 0.5


def is_overlap(w0, h0, width0, height0, w1, h1, width1, height1):
    return is_overlap_1d(w0, width0, w1, width1) and is_overlap_1d(h0, height0, h1, height1)


def is_overlap_1d(w0, width0, w1, width1):
    max_0 = w0 + width0 / 2
    min_0 = w0 - width0 / 2
    max_1 = w1 + width1 / 2
    min_1 = w1 - width1 / 2
    return not (max_0 < min_1 or min_0 > max_1)


def is_detected_agent_in_dead_zone(width, height, box_center, box_width, box_height):
    box_x, box_y = box_center

    x_in_deadzone = not is_x_cord_in_active_zone(box_x, width) or \
                    not is_x_cord_in_active_zone(box_x - box_width * 0.5, width) or \
                    not is_x_cord_in_active_zone(box_x + box_width * 0.5, width)

    if x_in_deadzone:
        return x_in_deadzone

    y_in_deadzone = not is_y_cord_in_active_zone(box_y, height) or \
                    not is_y_cord_in_active_zone(box_y - box_height * 0.5, height) or \
                    not is_y_cord_in_active_zone(box_y + box_height * 0.5, height)

    return y_in_deadzone


def is_x_cord_in_active_zone(x, width):
    return width * X_BOUNDARY_THRESHOLD_LEFT <= x <= width * (1 - X_BOUNDARY_THRESHOLD_RIGHT)


def is_y_cord_in_active_zone(y, height):
    return height * Y_BOUNDARY_THRESHOLD_TOP <= y <= height * (1 - Y_BOUNDARY_THRESHOLD_BOTTOM)


def alpha_shape(points, alpha):
    if len(points) < 4:
        return geometry.MultiPoint(list(points)).convex_hull
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def add_edge(edges, edge_points, coords, i, j):
    if (i, j) in edges or (j, i) in edges:
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])


def get_video_file_artifact_file_name(video_file_name, artifact_suffix_string):
    video_file_path = os.path.dirname(video_file_name)
    video_file_basename = os.path.basename(video_file_name)

    if ".mp4" in video_file_basename:
        artifact_file_name = video_file_basename.replace(".mp4", artifact_suffix_string)
    elif ".MP4" in video_file_basename:
        artifact_file_name = video_file_basename.replace(".MP4", artifact_suffix_string)
    else:
        print("[ERROR] utils: wrong file extension for video file %s" % video_file_name)

    return video_file_path, artifact_file_name


def compute_fps(num_times, total_time):
    if num_times > 0:
        t = total_time / num_times
        fps = int(1 / t)
    else:
        t = 0
        fps = 0
    return t, fps


def calculate_principal_axis(mask, scale_width_det_to_tracking, scale_height_det_to_tracking, sampling_occupancy=0.5):
    """
    this method below is only called when a DNN detection happens
    """
    try:
        y, x = np.nonzero(mask)
        num_samples = len(x)
        random_down_sampled_indicies = np.random.choice(num_samples, int(np.floor(sampling_occupancy * num_samples)),
                                                        replace=False)
        y = y[random_down_sampled_indicies]
        x = x[random_down_sampled_indicies]

        x = x - np.mean(x)
        y = y - np.mean(y)
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]

        # eigenvalues from large to small
        principal_axis_v1 = evecs[:, sort_indices[0]]  # eigenvector with largest eigenvalue,
        principal_axis_v2 = evecs[:, sort_indices[1]]

        x_v1, y_v1 = principal_axis_v1
        x_v2, y_v2 = principal_axis_v2

        # transform from detection space to tracking space
        x_v1 *= scale_width_det_to_tracking
        x_v2 *= scale_width_det_to_tracking
        y_v1 *= scale_height_det_to_tracking
        y_v2 *= scale_height_det_to_tracking

        principal_axis_v1 = np.array([x_v1, y_v1])
        principal_axis_v2 = np.array([x_v2, y_v2])
        principal_axis_v1 /= np.linalg.norm(principal_axis_v1)
        principal_axis_v2 /= np.linalg.norm(principal_axis_v2)

        x_v1, y_v1 = principal_axis_v1
        x_v2, y_v2 = principal_axis_v2

        index_x_max = np.argmax(x)
        index_x_min = np.argmin(x)
        index_y_max = np.argmax(y)
        index_y_min = np.argmin(y)

        x_max_point_proj_larger_axis, x_max_point_proj_smaller_axis = \
            compute_eigenaxis_points(index_x_max, x, y, principal_axis_v1, principal_axis_v2,
                                     scale_width_det_to_tracking, scale_height_det_to_tracking)

        x_min_point_proj_larger_axis, x_min_point_proj_smaller_axis = \
            compute_eigenaxis_points(index_x_min, x, y, principal_axis_v1, principal_axis_v2,
                                     scale_width_det_to_tracking, scale_height_det_to_tracking)

        y_max_point_proj_larger_axis, y_max_point_proj_smaller_axis = \
            compute_eigenaxis_points(index_y_max, x, y, principal_axis_v1, principal_axis_v2,
                                     scale_width_det_to_tracking, scale_height_det_to_tracking)

        y_min_point_proj_larger_axis, y_min_point_proj_smaller_axis = \
            compute_eigenaxis_points(index_y_min, x, y, principal_axis_v1, principal_axis_v2,
                                     scale_width_det_to_tracking, scale_height_det_to_tracking)

        larger_axis_max = max(x_max_point_proj_larger_axis, x_min_point_proj_larger_axis, y_max_point_proj_larger_axis,
                           y_min_point_proj_larger_axis)
        larger_axis_min = min(x_max_point_proj_larger_axis, x_min_point_proj_larger_axis, y_max_point_proj_larger_axis,
                              y_min_point_proj_larger_axis)

        larger_range = (larger_axis_max - larger_axis_min) * 0.5

        smaller_axis_max = max(x_max_point_proj_smaller_axis, x_min_point_proj_smaller_axis,
                            y_max_point_proj_smaller_axis, y_min_point_proj_smaller_axis)

        smaller_axis_min = min(x_max_point_proj_smaller_axis, x_min_point_proj_smaller_axis,
                               y_max_point_proj_smaller_axis, y_min_point_proj_smaller_axis)

        smaller_range = (smaller_axis_max - smaller_axis_min) * 0.5

        larger_principal_axis_point_1 = np.array([x_v1 * -larger_range, y_v1 * -larger_range])
        larger_principal_axis_point_2 = np.array([x_v1 * larger_range, y_v1 * larger_range])

        smaller_principal_axis_point_1 = np.array([x_v2 * -smaller_range, y_v2 * -smaller_range])
        smaller_principal_axis_point_2 = np.array([x_v2 * smaller_range, y_v2 * smaller_range])

        larger_principal_axis_vector = larger_principal_axis_point_2 - larger_principal_axis_point_1
        larger_principal_axis_vector = larger_principal_axis_vector / np.linalg.norm(larger_principal_axis_vector)

        smaller_principal_axis_vector = smaller_principal_axis_point_2 - smaller_principal_axis_point_1
        smaller_principal_axis_vector = smaller_principal_axis_vector / np.linalg.norm(smaller_principal_axis_vector)

        offset_corner_1 = larger_principal_axis_point_1 + smaller_principal_axis_point_1
        offset_corner_2 = larger_principal_axis_point_2 + smaller_principal_axis_point_1
        offset_corner_3 = larger_principal_axis_point_2 + smaller_principal_axis_point_2
        offset_corner_4 = larger_principal_axis_point_1 + smaller_principal_axis_point_2

        return larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1, \
            smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector, \
            offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4

    except Exception as e:
        logging.warning("utils: %s" % str(e))
        return None


def compute_eigenaxis_points(index, x, y, axis_v1, axis_v2, scale_width_det_to_tracking, scale_height_det_to_tracking):
    point = np.array([x[index] * scale_width_det_to_tracking, y[index] * scale_height_det_to_tracking])
    projection_axis_v1 = np.dot(point, axis_v1) / np.linalg.norm(axis_v1)
    projection_axis_v2 = np.dot(point, axis_v2) / np.linalg.norm(axis_v2)
    return projection_axis_v1, projection_axis_v2


def connect_to_ngsim_database(cloud_country_code):
    # Note: country code has already been convented to lower case letters
    assert cloud_country_code == "us" or cloud_country_code == "cn", "Country code %s wrong!" % cloud_country_code
    if cloud_country_code == "us":
        # remote one AWS US
        address = 'postgresql://rrusers:w8bxYGr7GZW$}z@rr-sample-data.cueq0qsvhbqr.us-west-2.rds.amazonaws.com:5432/BehavioralData'

    elif cloud_country_code == "cn":
        # remote one AWS China
        address = 'postgresql://rrusers:w8bxYGr7GZW$}z@rr-sample-data.cl0nv25kbusm.rds.cn-north-1.amazonaws.com.cn:5432/BehavioralData'
    else:
        logging.error("utils: wrong country code %s given!" % cloud_country_code)
        return None, None

    engine = create_engine(address)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    return connection, cursor


def get_calib_string_in_database(conn, cursor, user_id, calib_id):
    if conn is None or cursor is None:
        return ""

    query = "SELECT info FROM user_uploaded_calib WHERE info ->> 'UserID' = '%s' AND id = %s " % (user_id, calib_id)
    cursor.execute(query)
    rows = cursor.fetchall()

    if len(rows) <= 0:
        logging.info("utils: cannot find calib id %s for user %s" % (calib_id, user_id))
        return None

    json_string = str(rows[0][0])

    # Note for some reason json module requires key and values to be wrapped by double quotes only....
    json_string = json_string.replace("\'", "\"")
    return json.loads(json_string)


def calculate_mva(window_start_offset, window_size, list_values):
    """
    you need to provide the moving window starts from where, and how large window you would like to have
    """
    window_end = min(window_start_offset + window_size, -1)
    ave, c = 0., 0

    for w in list_values[window_start_offset:window_end]:
        ave += w
        c += 1

    if c <= 0:
        return list_values[-1]

    ave = ave / c

    return ave


def get_user_keyboard_input(key):
    is_quit, is_move_to_next_frame, user_defined_class = False, False, ""

    key = key & 0xFF

    if key == ord('q'):
        is_quit = True

    if key == ord('n'):
        is_move_to_next_frame = True

    if key == ord('c'):
        user_defined_class = 'Car'
    if key == ord('p'):
        user_defined_class = 'Pedestrian'
    if key == ord('m'):
        user_defined_class = 'Motorcycle'
    if key == ord('t'):
        user_defined_class = 'Truck'
    if key == ord('a'):
        user_defined_class = 'AV'
    if key == ord('b'):
        user_defined_class = 'Bicycle'
    if key == ord('B'):
        user_defined_class = 'Bus'

    if len(user_defined_class) > 0:
        assert user_defined_class in MAP_CATEGORY_NAME_SEMANTIC_CLASS,\
            "[ERROR] %s not in MAP_CATEGORY_NAME_SEMANTIC_CLASS" % user_defined_class

    return is_quit, is_move_to_next_frame, user_defined_class
