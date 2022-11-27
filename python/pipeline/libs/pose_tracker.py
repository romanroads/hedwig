import numpy as np
import cv2

import pipeline.utils.pose_flow as pf

from constant_values import FEATURE_PARAMS, PROCESSED_PREDICTIONS_BOX_WIDTH_NAME, PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME,\
    PROCESSED_PREDICTIONS_CENTER_NAME


class PoseTracker:
    def __init__(self, link_len=100, num=7, mag=30, match=0.2, orb_features=1000):
        self.frame_tracks = []
        self.combo_masks = []
        self.last_pid = 0
        self.link_len = link_len
        self.num = num
        self.mag = mag
        self.match = match
        self.orb_features = orb_features

        self.orb = cv2.ORB_create(nfeatures=orb_features, scoreType=cv2.ORB_FAST_SCORE)

        # FLANN parameters
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=12,
                            key_size=12,
                            multi_probe_level=2)
        search_params = dict(checks=100)  # or pass empty dictionary
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def track_gftt(self, frame_gray):
        tracking_feature_points = cv2.goodFeaturesToTrack(frame_gray, mask=self.mask_tracking_space,
                                                         **FEATURE_PARAMS)
        pass

    def track(self, frame, color_mask_tracking, keypoints, scores, processed_predictions):
        pe_tracks = []
        pose_pids = []
        weights = [1, 2, 1, 2, 0, 0]
        weights_fff = [0, 1, 0, 1, 0, 0]

        if len(keypoints) == 0:
            return pose_pids

        for (i, instance_keypoints) in enumerate(keypoints):
            processed_instance = processed_predictions[i]
            b_center = processed_instance[PROCESSED_PREDICTIONS_CENTER_NAME]
            b_w = processed_instance[PROCESSED_PREDICTIONS_BOX_WIDTH_NAME]
            b_h = processed_instance[PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME]

            xmin, xmax, ymin, ymax = float(b_center[0] - b_w * 0.5), float(b_center[0] + b_w * 0.5),\
                float(b_center[1] - b_h * 0.5), float(b_center[1] + b_h * 0.5)

            pe_track = {"box_pos": [xmin, xmax, ymin, ymax],
                        "box_score": scores[i],
                        "keypoints_pos": instance_keypoints[:, 0:2], "keypoints_score": instance_keypoints[:, -1],
                        "det_index": i}

            # init tracking info of the first frame
            if len(self.frame_tracks) == 0:
                pe_track["new_pid"] = i
                pe_track["match_score"] = 0
                self.last_pid = i
            pe_tracks.append(pe_track)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.frame_tracks.append((frame, pe_tracks))
        self.combo_masks.append(color_mask_tracking)

        if len(self.frame_tracks) == 1:  # Is it the first frame?
            for i in range(len(keypoints)):
                (x1, x2, y1, y2) = pe_tracks[i]["box_pos"]
                pose_pids.append({
                    "pid": pe_tracks[i]["new_pid"],
                    "score": pe_tracks[i]["match_score"],
                    "box": np.array([x1, y1, x2, y2]),
                    "det_index": pe_tracks[i]["det_index"]
                })
            return pose_pids

        # Get previous frame data
        prev_frame, prev_pe_track = self.frame_tracks[-2]
        prev_mask = self.combo_masks[-2]

        # Match ORB descriptor vectors for current and previous frame
        # with a FLANN (Fast Library for Approximate Nearest Neighbors) based matcher
        all_cors = pf.orb_matching(prev_frame, frame, prev_mask, color_mask_tracking, self.orb, self.flann_matcher)

        # Stack all already tracked agents's info together
        curr_all_pids, curr_all_pids_fff = pf.stack_all_pids(self.frame_tracks, self.last_pid, self.link_len)

        # Hungarian matching algorithm
        match_indexes, match_scores = pf.best_matching_hungarian(
            all_cors, curr_all_pids, curr_all_pids_fff, self.frame_tracks[-1], weights, weights_fff, self.num, self.mag)

        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > self.match:
                self.frame_tracks[-1][1][pid2]["new_pid"] = curr_all_pids[pid1]["new_pid"]
                self.last_pid = max(self.last_pid, self.frame_tracks[-1][1][pid2]["new_pid"])
                self.frame_tracks[-1][1][pid2]["match_score"] = match_scores[pid1][pid2]

        # add the untracked new person
        for next_pid in range(len(self.frame_tracks[-1][1])):
            if "new_pid" not in self.frame_tracks[-1][1][next_pid]:
                self.last_pid += 1
                self.frame_tracks[-1][1][next_pid]["new_pid"] = self.last_pid
                self.frame_tracks[-1][1][next_pid]["match_score"] = 0

        for i in range(len(self.frame_tracks[-1][1])):
            (x1, x2, y1, y2) = self.frame_tracks[-1][1][i]["box_pos"]
            pose_pids.append({
                "pid": self.frame_tracks[-1][1][i]["new_pid"],
                "score": self.frame_tracks[-1][1][i]["match_score"],
                "box": np.array([x1, y1, x2, y2]),
                "det_index": self.frame_tracks[-1][1][i]["det_index"]
            })

        return pose_pids




