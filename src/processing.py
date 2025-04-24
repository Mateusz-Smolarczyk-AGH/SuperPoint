import numpy as np
import cv2
import torch
import superpoint_pytorch
import os
from scipy.spatial.transform import Rotation as R
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PinholeCamera:
	def __init__(self, fx, fy, cx, cy, 
				d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0):
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.d = np.array([d0, d1, d2, d3, d4])

class Feature_detection():
    def __init__(self, image_size):
        self.input_image = None
        self.input = None
        # self.model = superpoint_pytorch.SuperPoint_short(detection_threshold=0.005, nms_radius=5).eval()
        # self.model.load_state_dict(torch.load("weights/superpoint_v6_from_tf.pth"))
        self.model = superpoint_pytorch.SuperPointNet(detection_threshold=0.005, nms_radius=5).eval()
        self.model.load_state_dict(torch.load("weights/superpoint_v1.pth"))
        self.image_size = image_size

    def get_input(self, image) -> None:
        """
        Preprocess raw image from camera or dataset.
        """
        img = cv2.resize(image, self.image_size)
        self.input_image = img.copy()
        self.input = img.mean(-1) / 255

    def process_Superpoint(self) -> None:
        """
        Process NN and short post processing.
        """
        start = time.perf_counter()
        input_tensor = torch.from_numpy(self.input[None, None]).float()
        scores, descriptors_dense = self.model(input_tensor)
        net = time.perf_counter()

        pred_th_1 = self.post_processing_short(scores, descriptors_dense)
        descriptors = pred_th_1['descriptors'][0].cpu().detach().numpy().astype(np.float32)
        points_th = pred_th_1['keypoints'][0]
        keypoints_np = np.array(points_th)
        keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        post = time.perf_counter()

        return keypoints, descriptors, net-start, post-net

    def post_processing_short(self, scores, descriptors_dense):
        """
        Realise post processing including:
        * discarding points near image border
        * converting keypoints
        * nms
        """
        conf = self.model.conf
        b = scores.shape[0]
        scores = superpoint_pytorch.batched_nms(scores, 4)

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
    def __init__(self, image_size, start_R, start_t, cam: PinholeCamera):
        self.keypoints = {"past": None,
                          "present": None}

        self.descriptors = {"past": None,
                          "present": None}
        self.matches = None
        self.feature_detection = Feature_detection(image_size)
        self.R_total = np.eye(3)
        self.start_R = start_R
        self.t_total = np.zeros((3,1))
        self.start_t = start_t
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        self.d = cam.d
        self.trajectory = [self.t_total.flatten().tolist()]
        r = R.from_matrix(start_R)
        angles = r.as_euler('zyx', degrees=True)  # yaw, pitch, roll
        self.R_list = [angles]

    def match_descriptors(self):
        """ 
        Match the keypoints with the warped_keypoints with nearest neighbor search
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

    def compute_homography(self, matched_kp1, matched_kp2):
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

    def rotation_angle(self, R):
        R_diff = R @ self.R_total.T
        angle = np.arccos((np.trace(R_diff) - 1) / 2)
        return np.degrees(angle)
    
    def pose_estimation(self, points1, points2):
        E, mask = cv2.findEssentialMat(points2, points1, focal=self.focal, pp=self.pp,  method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R_diff, t, mask = cv2.recoverPose(E, points2[mask.ravel().astype(bool)], points1[mask.ravel().astype(bool)], focal=self.focal, pp=self.pp)
        angles_change = R.from_matrix(R_diff).as_euler('zyx', degrees=True)
        if np.any(np.abs(angles_change) > 15):
            return 1
        self.t_total = self.t_total + self.R_total.dot(t)
        self.R_total = self.R_total.dot(R_diff)
        return 0

    def compute_first_image(self, image):
        self.feature_detection.get_input(image)
        self.keypoints["past"], self.descriptors["past"], _, _ = self.feature_detection.process_Superpoint()

    def compute_pipeline(self, image):
        start = time.perf_counter()
        self.feature_detection.get_input(image)
        pre = time.perf_counter()
        self.keypoints["present"], self.descriptors["present"], net, post = self.feature_detection.process_Superpoint()
        match = time.perf_counter()
        m_kp1, m_kp2 = self.match_descriptors()
        H, inliers = self.compute_homography(m_kp1, m_kp2)
        self.matches = np.array(self.matches)[inliers.astype(bool)].tolist()
        pts1 = np.float32([self.keypoints["past"][match.queryIdx].pt for match in self.matches])  # shape (N, 2)
        pts2 = np.float32([self.keypoints["present"][match.trainIdx].pt for match in self.matches])  # shape (N, 2)
        end = time.perf_counter()
        result = self.pose_estimation(pts1, pts2)
        r_global = self.start_R.dot(self.R_total)
        r = R.from_matrix(r_global)
        angles = r.as_euler('zyx', degrees=True)  # yaw, pitch, roll
        self.R_list.append(angles)
        if result == 0:
            trac = self.start_R.dot(self.t_total) + self.start_t
            self.trajectory.append(trac.flatten().tolist())
            self.show_arrows()
        
        return result, (pre-start, net, post, end-match, len(self.matches))
