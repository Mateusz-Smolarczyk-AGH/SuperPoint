import numpy as np
import cv2
import torch
import superpoint_pytorch, superglue
import os
from scipy.spatial.transform import Rotation as R
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def crop_center(image, target_size):
    h, w = image.shape[:2]
    new_w, new_h = target_size

    # Oblicz przesunięcia
    x_start = (w - new_w) // 2
    y_start = (h - new_h) // 2

    # Wytnij wycinek
    cropped = image[y_start:y_start + new_h, x_start:x_start + new_w]
    return cropped, (x_start, y_start)


class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = np.array([d0, d1, d2, d3, d4])

class Feature_detection():
    def __init__(self, image_size, camera, database):
        self.input_image = None
        self.input = None
        # self.model = superpoint_pytorch.SuperPoint_short(detection_threshold=0.005, nms_radius=5).eval()
        # self.model.load_state_dict(torch.load("weights/superpoint_v6_from_tf.pth", weights_only=True))
        self.model = superpoint_pytorch.SuperPointNet(
            detection_threshold=0.005, nms_radius=5
        ).eval()
        self.model.load_state_dict(torch.load("weights/superpoint_v1.pth", weights_only=True))
        self.database = database
        self.image_size = image_size
        self.K_l = np.array([camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0]).reshape(3, 3)
        self.d_l = camera.d

    def get_input(self, img) -> None:
        """
        Preprocess raw image from camera or dataset.
        """
        if self.database == "tum":
            img = cv2.resize(img, (self.image_size[0] + 32, self.image_size[1] + 16))
            img = cv2.undistort(img, self.K_l, self.d_l)
            img = img[8:-8, 16:-16, :]
        else:
            # img, offset = crop_center(img, self.image_size)

            img = cv2.resize(img, self.image_size)

        self.input_image = img.copy()
        self.input = img.mean(-1) / 255

    def process_Superpoint(self) -> tuple:
        """
        Process NN and short post processing.
        """
        start = time.perf_counter()
        input_tensor = torch.from_numpy(self.input[None, None]).float()
        scores, descriptors_dense = self.model(input_tensor)
        net = time.perf_counter()

        predictions = self.post_processing_short(scores, descriptors_dense)
        post = time.perf_counter()

        return predictions, net-start, post-net

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
                k, s = superpoint_pytorch.select_top_k_keypoints(
                    k, s, conf.max_num_keypoints
                )
            d = superpoint_pytorch.sample_descriptors(
                k[None], descriptors_dense[i, None], 2 ** (len(conf.channels) - 2)
            )
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }

class VisualOdometry():
    def __init__(self, image_size, start_R, start_t, cam: PinholeCamera, matching_type='bf', database='tum'):
        # self.keypoints = {"past": None,
        #                   "present": None}

        # self.descriptors = {"past": None,
        #                   "present": None}
        self.past_predictions = {}
        self.present_predictions = {}
        self.matches = None
        self.feature_detection = Feature_detection(image_size, cam, database)
        self.R_total = np.eye(3)
        self.start_R = start_R
        self.t_total = np.zeros((3, 1))
        self.start_t = start_t
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.K = np.array(
            [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]]
        )
        self.d = cam.d
        self.trajectory = [self.start_t.flatten().tolist()]
        r = R.from_matrix(start_R)
        angles = r.as_euler("zyx", degrees=True)  # yaw, pitch, roll
        self.R_list = [angles]
        self.matching_type = matching_type
        if matching_type == "SuperGlue":
            config = {
            'weights': 'indoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
            'superglue_weights': r"weights\superglue_indoor.pth",
            'shape': image_size}
            self.matcher = superglue.SuperGlue(config).eval()

    def match_descriptors_bf(self):
        """ 
        Match the keypoints with the warped_keypoints with nearest neighbor search
        """
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)   # or pass empty dictionary
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # self.matches = flann.knnMatch(self.past_predictions["descriptors"], self.present_predictions["descriptors"] ,k=1)
        # self.matches = np.array([m[0] for m in self.matches])

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        kp1 = self.past_predictions["keypoints"]
        kp2 = self.present_predictions["keypoints"]
        self.matches = bf.match(self.past_predictions["descriptors"], self.present_predictions["descriptors"])
        # self.matches = sorted(self.matches, key = lambda x:x.distance)
        # if len(self.matches) > 200:
        #     self.matches = self.matches[:200]
        matches_idx = np.array([m.queryIdx for m in self.matches])
        m_kp1 = [kp1[idx] for idx in matches_idx]
        matches_idx = np.array([m.trainIdx for m in self.matches])
        m_kp2 = [kp2[idx] for idx in matches_idx]

        return m_kp1, m_kp2

    def superglue_match(self):
        # Preprocess the data
        keys = ['keypoints', 'keypoint_scores', 'descriptors']
        last_data = {k+'0': self.past_predictions['raw'][k] for k in keys}
        prediction_data = {k+'1': v for k, v in self.present_predictions['raw'].items()}
        data = {**last_data, **prediction_data}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = self.matcher(data)
        
        return {**data, **pred}


    def compute_homography(self, matched_kp1, matched_kp2):
        if isinstance(matched_kp1[0], cv2.KeyPoint):
            matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
            matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
        else:
            matched_pts1 = matched_kp1
            matched_pts2 = matched_kp2

        # Estimate the homography between the matches using RANSAC
        H, inliers = cv2.findHomography(matched_pts1, matched_pts2, cv2.RANSAC)
        inliers = inliers.flatten()
        return H, inliers
    
    def show_arrows(self, points_from, points_to, color=(0, 255, 0), thickness=1, tipLength=0.2):
        """
        Rysuje strzałki od punktów 'points_from' do 'points_to' na obrazie.
        
        Parameters:
            image (np.ndarray): Obraz wejściowy (modyfikowany w miejscu).
            points_from (np.ndarray): Punkty początkowe, shape (N, 2)
            points_to (np.ndarray): Punkty końcowe, shape (N, 2)
            color (tuple): Kolor strzałek w BGR.
            thickness (int): Grubość linii.
            tipLength (float): Długość grotu strzałki (0–1).
        """
        for pt1, pt2 in zip(points_from, points_to):
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            cv2.arrowedLine(self.feature_detection.input_image, pt1, pt2, color, thickness, tipLength=tipLength)

    def pose_estimation(self, points1, points2, abs_scale):
        E, mask = cv2.findEssentialMat(points2, points1, focal=self.focal, pp=self.pp,  method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # best_num_inliers = 0
        # for _E in np.split(E, len(E) / 3):
        #     n, R_temp, t_temp, _ = cv2.recoverPose(
        #         _E, points2, points1, focal=self.focal, pp=self.pp, mask=mask)
        #     if n > best_num_inliers:
        #         best_num_inliers = n
        #         R_diff, t = R_temp, t_temp
        #         # ret = (R, t[:, 0], mask.ravel() > 0)
        points, R_diff, t, mask = cv2.recoverPose(E, points2[mask.ravel().astype(bool)], points1[mask.ravel().astype(bool)], focal=self.focal, pp=self.pp)
        
        angles_change = R.from_matrix(R_diff).as_euler('zyx', degrees=True)
        if np.any(np.abs(angles_change) > 15):
            return 1
        # abs_scale  = 1
        self.t_total = self.t_total + abs_scale*self.R_total.dot(t)
        self.R_total = self.R_total.dot(R_diff)
        return 0

    def compute_first_image(self, image):
        self.feature_detection.get_input(image)
        self.past_predictions['raw'], _, _ = self.feature_detection.process_Superpoint()
        self.past_predictions['descriptors'] = self.past_predictions['raw']['descriptors'][0].cpu().detach().numpy().astype(np.float32)
        points_th = self.past_predictions['raw']['keypoints'][0]
        keypoints_np = np.array(points_th)
        self.past_predictions['keypoints'] = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        # gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        # self.past_predictions['keypoints'], self.past_predictions['descriptors'] = sift.detectAndCompute(gray,None)

    def compute_pipeline(self, image, pos_cur=None, pos_prev=None):
        start = time.perf_counter()
        #preprocessing
        self.feature_detection.get_input(image)
        pre = time.perf_counter()
        #superPoint
        self.present_predictions['raw'], net, post = self.feature_detection.process_Superpoint()
        # net = 0
        # post = 0
        match = time.perf_counter()
        #matching
        self.present_predictions['descriptors'] = self.present_predictions['raw']['descriptors'][0].cpu().detach().numpy().astype(np.float32)
        points_th = self.present_predictions['raw']['keypoints'][0]
        keypoints_np = np.array(points_th)
        self.present_predictions['keypoints'] = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        # gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        # self.present_predictions['keypoints'], self.present_predictions['descriptors'] = sift.detectAndCompute(gray,None)

        if self.matching_type == "bf":      
            m_kp1, m_kp2 = self.match_descriptors_bf()
            H, inliers = self.compute_homography(m_kp1, m_kp2)
            self.matches = np.array(self.matches)[inliers.astype(bool)].tolist()
            pts1 = np.float32([self.past_predictions['keypoints'][match.queryIdx].pt for match in self.matches])  # shape (N, 2)
            pts2 = np.float32([self.present_predictions['keypoints'][match.trainIdx].pt for match in self.matches])  # shape (N, 2)
        if self.matching_type == "SuperGlue":
            sg_matching = self.superglue_match()
            sg_kpts0 = self.past_predictions['raw']['keypoints'][0].cpu().numpy()
            sg_kpts1 = sg_matching['keypoints1'][0].cpu().numpy()
            sg_matches = sg_matching['matches0'][0].cpu().numpy()
            sg_confidence = sg_matching['matching_scores0'][0].detach().cpu().numpy()

            sg_valid = sg_matches > -1
            m_kp1 = sg_kpts0[sg_valid]
            m_kp2 = sg_kpts1[sg_matches[sg_valid]]        
            H, inliers = self.compute_homography(m_kp1, m_kp2)
            pts1 = m_kp1[inliers.astype(bool)]
            pts2 = m_kp2[inliers.astype(bool)]
        end = time.perf_counter()
        #pose estimation
        if pos_cur is None:
            abs_scale = 1
        else:
            abs_scale = np.linalg.norm(pos_cur - pos_prev)
        # abs_scale = np.sqrt((pos_cur[0] - pos_prev[0])*(pos_cur[0] - pos_prev[0]) + (pos_cur[1] - pos_prev[1])*(pos_cur[1] - pos_prev[1]) + (pos_cur[1] - pos_prev[1])*(pos_cur[1] - pos_prev[1]))
        result = self.pose_estimation(pts1, pts2, abs_scale)
        r_global = self.start_R.dot(self.R_total)
        r = R.from_matrix(r_global)
        angles = r.as_euler("zyx", degrees=True)  # yaw, pitch, roll
        self.R_list.append(angles)
        # trac = self.t_total + self.start_t
        trac = self.start_R.dot(self.t_total) + self.start_t
        self.trajectory.append(trac.flatten().tolist())
        self.show_arrows(pts1, pts2)
        
        return result, (pre-start, net, post, end-match, len(pts1))
