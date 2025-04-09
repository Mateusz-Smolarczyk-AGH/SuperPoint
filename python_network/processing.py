import numpy as np
import cv2
import torch
import superpoint_pytorch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Feature_detection():
    def __init__(self, image_size):
        self.input_image = None
        self.input = None
        self.model = superpoint_pytorch.SuperPoint_short(detection_threshold=0.005, nms_radius=5).eval()
        self.model.load_state_dict(torch.load("weights/superpoint_v6_from_tf.pth"))
        self.image_size = image_size

    def get_input(self, image) -> None:
        """
        Preprocess raw image from camera or dataset.
        """
        img = cv2.resize(image, self.img_size)
        self.input_image = img.copy()
        self.input = img.mean(-1) / 255

    def process_Superpoint(self) -> None:
        """
        Process NN and short post processing.
        """
        input_tensor = torch.from_numpy(self.input[None, None]).float()
        scores, descriptors_dense = self.model(input_tensor)
        pred_th_1 = self.post_processing_short(self, scores, descriptors_dense)
        descriptors = pred_th_1['descriptors'][0].cpu().detach().numpy().astype(np.float32)
        points_th = pred_th_1['keypoints'][0]
        keypoints_np = np.array(points_th)  # Konwersja do NumPy
        keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        return keypoints, descriptors

    def post_processing_short(self, scores, descriptors_dense):
        """
        Realise post processing including:
        * discarding points near image border
        * converting keypoints
        * nms
        """
        conf = self.model.conf
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

class VisualOdometry():
    def __init__(self, image_size, start_R, start_t, K):
        self.keypoints = {"past": None,
                          "present": None}
        self.matched_keypoints = {"past": None,
                          "present": None}
        self.descriptors = {"past": None,
                          "present": None}
        self.matches = None
        self.feature_detection = Feature_detection(image_size)
        self.total_R = start_R
        self.total_t = start_t
        self.K = K
        self. trajectory = [start_t]

    def match_descriptors(self):
        """ Match the keypoints with the warped_keypoints with nearest neighbor search
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        kp1 = self.keypoints["past"]
        kp2 = self.keypoints["present"]
        self.matches = bf.match(self.descriptors["past"], self.descriptors["present"])
        matches_idx = np.array([m.queryIdx for m in self.matches])
        m_kp1 = [kp1[idx] for idx in matches_idx]
        matches_idx = np.array([m.trainIdx for m in self.matches])
        m_kp2 = [kp2[idx] for idx in matches_idx]

        return m_kp1, m_kp2

    def compute_homography(matched_kp1, matched_kp2):
        matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
        matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

        # Estimate the homography between the matches using RANSAC
        H, inliers = cv2.findHomography(matched_pts1,
                                        matched_pts2,
                                        cv2.RANSAC)
        inliers = inliers.flatten()
        return H, inliers
    
    def show_arrows(self):
        for match in self.matches:
            pt1 = tuple(map(int, self.keypoints["past"][match.queryIdx].pt))
            pt2 = tuple(map(int, self.keypoints["present"][match.trainIdx].pt))
            cv2.arrowedLine(self.feature_detection.input_image, pt2, pt1, (0, 255, 0), 1, tipLength=0.2)

    def pose_estimation(self):
        E, mask = cv2.findEssentialMat(self.matched_keypoints["past"], self.matched_keypoints["present"], self.K,  method=cv2.RANSAC, prob=0.999, threshold=3.0)
        points, R, t, mask = cv2.recoverPose(E, self.matched_keypoints["past"], self.matched_keypoints["present"], self.K)
        self.R_total = self.R_total @ R
        self.t_total = self.t_total + self.R_total @ t

    def compute_pipeline(self, image):
        self.feature_detection.get_input(image)
        self.keypoints["present"], self.descriptors["present"] = self.feature_detection.process_Superpoint()
        m_kp1, m_kp2 = self.match_descriptors()
        H, inliers = self.compute_homography(m_kp1, m_kp2)
        self.matches = np.array(self.matches)[inliers.astype(bool)].tolist()
        for match in self.matches:
            self.matched_keypoints["past"].append(self.keypoints["past"][match.queryIdx].pt)
            self.matched_keypoints["present"].append(self.keypoints["present"][match.trainIdx].pt)
        self.pose_estimation()
        self.trajectory.append(self.total_t.flatten().tolist())
