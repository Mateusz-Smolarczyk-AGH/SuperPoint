import numpy as np
import cv2
import torch

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

detection_thresh = 0.005
nms_radius = 5

import superpoint_pytorch

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


def show_comparison(image1_path, image2_path, model, nettype='Normal'):
    img_size = (300, 200)
    image1, img1_orig = preprocess_image(image1_path, img_size)
    image2, img2_orig = preprocess_image(image2_path, img_size)

    # Run inference for both images
    images = [image1, image2]
    keypoints_list = []
    desc_list = []

    for image in images:
        with torch.no_grad():
            if nettype=='Normal':
                pred_th_1 = model({'image': torch.from_numpy(image[None, None]).float()})
            else:
                scores, descriptors_dense = model(torch.from_numpy(image[None, None]).float())
                pred_th_1 = post_processing_short(scores, descriptors_dense, model.conf)
        # Extract descriptors
        descriptors = pred_th_1['descriptors'][0]
        points_th = pred_th_1['keypoints'][0]
        keypoints_np = np.array(points_th)  # Konwersja do NumPy
        keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        keypoints_list.append(keypoints)
        desc_list.append(descriptors.cpu().detach().numpy().astype(np.float32)
)

    m_kp1, m_kp2, matches = match_descriptors(keypoints_list[0], desc_list[0], keypoints_list[1], desc_list[1])
    H, inliers = compute_homography(m_kp1, m_kp2)

    # Draw SuperPoint matches
    matches = np.array(matches)[inliers.astype(bool)].tolist()
    matched_img = cv2.drawMatches(img1_orig, keypoints_list[0], img2_orig, keypoints_list[1], matches,
                                    None, matchColor=(0, 255, 0),
                                    singlePointColor=(0, 0, 255))

    return matched_img

def test_on_HPatches(dir_name, nettype='Normal'):
    if nettype == 'Normal':
        model = superpoint_pytorch.SuperPoint(detection_threshold=detection_thresh, nms_radius=nms_radius).eval()
    else:
        model = superpoint_pytorch.SuperPoint_short(detection_threshold=detection_thresh, nms_radius=nms_radius).eval()
    model.load_state_dict(torch.load("weights/superpoint_v6_from_tf.pth"))
    datadir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data/hpatch-release/"))
    img_path1 = datadir + dir_name + "/1.ppm"
    image_list = []
    for i in range(2, 7):
        img_path2 = datadir + dir_name + f"/{i}.ppm"
        transformation = datadir + dir_name + f"/H_{1}_{i}"
        image_list.append(show_comparison(img_path1, img_path2, model, nettype))
        print(f"Progres: {int(((i-1)/5)*100)}%")
    h, w = image_list[0].shape[:2]
    resized_images = [cv2.resize(img, (w, h)) for img in image_list]
    vertical_stack = np.vstack(resized_images)
    cv2.imwrite("data/matched_image.png", vertical_stack)

test_on_HPatches("/i_kions", nettype="Unnormal")
model = torch.load("weights/superpoint_v6_from_tf.pth", map_location="cpu")
torch.save(model, "model_weights_legacy.pth", _use_new_zipfile_serialization=False)

