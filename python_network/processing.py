import numpy as np
import cv2
import torch
import superpoint_pytorch

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(torch.__version__)

detection_thresh = 0.005
nms_radius = 5


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches

def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1,
                                    matched_pts2,
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers

def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()
    image = img.mean(-1) / 255
    img_preprocessed = np.pad(image, [(0, int(np.ceil(s/8))*8 - s) for s in image.shape[:2]])

    return img_preprocessed, img_orig

def post_processing_short(scores, descriptors_dense, conf):
    b = scores.shape[0]
    scores = superpoint_pytorch.batched_nms(scores, conf.nms_radius)

    # Discard keypoints near the image borders
    if conf.remove_borders:
        pad = conf.remove_borders
        scores[:, :pad] = -1
        scores[:, :, :pad] = -1
        scores[:, -pad:] = -1
        scores[:, :, -pad:] = -1

    # Extract keypoints
    if b > 1:
        idxs = torch.where(scores > conf.detection_threshold)
        mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
    else:  # Faster shortcut
        scores = scores.squeeze(0)
        idxs = torch.where(scores > conf.detection_threshold)

    # Convert (i, j) to (x, y)
    keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
    scores_all = scores[idxs]

    keypoints = []
    scores = []
    descriptors = []
    for i in range(b):
        if b > 1:
            k = keypoints_all[mask[i]]
            s = scores_all[mask[i]]
        else:
            k = keypoints_all
            s = scores_all
        if conf.max_num_keypoints is not None:
            k, s = superpoint_pytorch.select_top_k_keypoints(k, s, conf.max_num_keypoints)
        d = superpoint_pytorch.sample_descriptors(k[None], descriptors_dense[i, None], 2 ** (len(conf.channels) - 2))
        keypoints.append(k)
        scores.append(s)
        descriptors.append(d.squeeze(0).transpose(0, 1))

    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "descriptors": descriptors,
    }
