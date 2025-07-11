from typing import Optional, Tuple
import numpy as np
import cv2
import torch
import superpoint_pytorch, superglue
import os
from scipy.spatial.transform import Rotation as R
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def crop_center(image, target_size) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop the central region of an image to the specified target size.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        target_size (tuple[int, int]): Target size as (width, height) of the cropped region.

    Returns:
        tuple:
            - cropped (np.ndarray): The cropped image.
            - (x_start, y_start) (tuple[int, int]): Coordinates of the top-left corner of the crop in the original image.
    """
    h, w = image.shape[:2]
    new_w, new_h = target_size

    x_start = (w - new_w) // 2
    y_start = (h - new_h) // 2

    cropped = image[y_start : y_start + new_h, x_start : x_start + new_w]
    return cropped, (x_start, y_start)


class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = np.array([d0, d1, d2, d3, d4])


class Feature_detection:
    def __init__(self, image_size, camera, database, superpoint_weights=None):
        self.input_image = None
        self.input = None
        # self.model = superpoint_pytorch.SuperPoint_short(detection_threshold=0.005, nms_radius=5).eval()
        # self.model.load_state_dict(torch.load("weights/superpoint_v6_from_tf.pth", weights_only=True))
        self.model = superpoint_pytorch.SuperPointNet(
            detection_threshold=0.005, nms_radius=5
        ).eval()
        self.model.load_state_dict(torch.load(superpoint_weights, weights_only=True))
        self.database = database
        self.image_size = image_size
        self.K_l = np.array(
            [camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0]
        ).reshape(3, 3)
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

    def process_Superpoint(self) -> tuple[dict[str, torch.Tensor], float, float]:
        """
        Process the SuperPoint neural network and perform short post-processing.

        Runs the forward pass through the model and extracts sparse keypoints 
        from dense predictions via post-processing.

        Returns:
            tuple:
                - predictions (dict[str, torch.Tensor]): Output from post-processing, 
                typically including keypoints, scores, and descriptors.
                - inference_time (float): Time in seconds for model inference.
                - postprocessing_time (float): Time in seconds for post-processing.
        """
        start = time.perf_counter()

        input_tensor = torch.from_numpy(self.input[None, None]).float()

        scores, descriptors_dense = self.model(input_tensor)
        net = time.perf_counter()

        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        softmax = time.perf_counter()

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * 8, w * 8
        )
        dn = torch.norm(descriptors_dense, p=2, dim=1) # Compute the norm.
        descriptors_dense = descriptors_dense.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        predictions = self.post_processing_short(scores, descriptors_dense)
        post = time.perf_counter()

        return predictions, net - start, post - net

    def post_processing_short(self, scores: torch.Tensor, descriptors_dense: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """
        Apply short post-processing to SuperPoint outputs.

        This includes:
            ** discarding keypoints near image borders
            ** converting (i, j) to (x, y) coordinates
            ** applying non-maximum suppression (NMS)
            ** selecting top keypoints
            ** sampling corresponding descriptors

        Parameters:
            scores (torch.Tensor): Heatmap tensor of keypoint scores, shape (B, H, W).
            descriptors_dense (torch.Tensor): Dense descriptors from SuperPoint, shape (B, C, H/8, W/8).

        Returns:
            dict[str, list[torch.Tensor]]: Dictionary containing:
                - "keypoints": list of (N_i, 2) tensors with 2D keypoint coordinates per image.
                - "keypoint_scores": list of (N_i,) tensors with scores per keypoint.
                - "descriptors": list of (N_i, D) tensors with corresponding descriptors.
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


class VisualOdometry:
    def __init__(
        self,
        config: dict,
        image_size: tuple[int, int],
        start_R: np.ndarray,
        start_t: np.ndarray,
        cam: PinholeCamera,
        matching_type: str = "bf",
        database: str = "tum",
        superpoint_weights: str = None,
        superglue_weights: str = None,
    ):
        """
        Visual odometry pipeline using either brute-force or SuperGlue feature matching.

        Parameters:
            config (dict): Configuration dictionary for parameters and thresholds.
            image_size (tuple[int, int]): Input image size (width, height).
            start_R (np.ndarray): Initial rotation matrix (3x3).
            start_t (np.ndarray): Initial translation vector (3x1).
            cam (PinholeCamera): Camera intrinsic and distortion model.
            matching_type (str): Feature matching method ("bf" or "SuperGlue").
            database (str): Dataset type (e.g., "tum").
            superpoint_weights (str, optional): Path to SuperPoint weights.
            superglue_weights (str, optional): Path to SuperGlue weights.
        """

        self.args = config
        # depth
        self.past_depth = None
        self.present_depth = None

        self.past_predictions = {}
        self.present_predictions = {}
        self.matches = None
        self.feature_detection = Feature_detection(
            image_size, cam, database, superpoint_weights
        )
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
        starts =self.start_R.dot(self.t_total) + self.start_t
        self.trajectory = [starts.flatten().tolist()]
        r = R.from_matrix(start_R)
        angles = r.as_euler("zyx", degrees=True)  # yaw, pitch, roll
        self.R_list = [angles]
        self.matching_type = matching_type
        if matching_type == "SuperGlue":
            config = {
                "weights": "indoor",
                "sinkhorn_iterations": 100,
                "match_threshold": 0.2,
                "superglue_weights": superglue_weights,
                "shape": image_size,
            }
            self.matcher = superglue.SuperGlue(config).eval()

    def match_descriptors_bf(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Match descriptors between two sets of keypoints using brute-force matching.

        Uses OpenCV's BFMatcher with L2 norm and crossCheck enabled to find 
        one-to-one nearest neighbor matches between descriptors from 
        `self.past_predictions` and `self.present_predictions`.

        Returns:
            tuple:
                - m_kp1 (list[np.ndarray]): List of matched keypoints from the past frame.
                - m_kp2 (list[np.ndarray]): List of corresponding matched keypoints from the present frame.
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
        self.matches = bf.match(
            self.past_predictions["descriptors"],
            self.present_predictions["descriptors"],
        )

        # self.matches = sorted(self.matches, key = lambda x:x.distance)
        # if len(self.matches) > 200:
        #     self.matches = self.matches[:200]
        matches_idx = np.array([m.queryIdx for m in self.matches])
        m_kp1 = [kp1[idx] for idx in matches_idx]
        matches_idx = np.array([m.trainIdx for m in self.matches])
        m_kp2 = [kp2[idx] for idx in matches_idx]

        return m_kp1, m_kp2

    def superglue_match(self) -> dict[str, torch.Tensor]:
        """
        Match the keypoints with the warped_keypoints with SupreGlue NN.
        """

        # Preprocess the data
        keys = ["keypoints", "keypoint_scores", "descriptors"]
        last_data = {k + "0": self.past_predictions["raw"][k] for k in keys}
        prediction_data = {
            k + "1": v for k, v in self.present_predictions["raw"].items()
        }
        data = {**last_data, **prediction_data}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = self.matcher(data)

        return {**data, **pred}

    def compute_homography(
        self,
        matched_kp1: list[cv2.KeyPoint] | np.ndarray,
        matched_kp2: list[cv2.KeyPoint] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate homography matrix from matched keypoints using RANSAC.

        Parameters:
            matched_kp1 (list[cv2.KeyPoint] | np.ndarray): Matched keypoints from the first image.
            matched_kp2 (list[cv2.KeyPoint] | np.ndarray): Corresponding matched keypoints from the second image.

        Returns:
            tuple:
                - H (np.ndarray): Estimated 3x3 homography matrix.
                - inliers (np.ndarray): Binary mask of inliers (1) and outliers (0) determined by RANSAC.
        """
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

    def show_arrows(
        self,
        points_from: np.ndarray,
        points_to: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 1,
        tipLength: float = 0.2
    ) -> None:
        """
        Draws arrows from 'points_from' to 'points_to' on the image.

        Parameters:
            image (np.ndarray): Input image (modified in place).
            points_from (np.ndarray): Starting points, shape (N, 2).
            points_to (np.ndarray): Ending points, shape (N, 2).
            color (tuple): Arrow color in BGR format.
            thickness (int): Line thickness.
            tipLength (float): Length of the arrow tip (range 0–1).
        """
        for pt1, pt2 in zip(points_from, points_to):
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            # draw points
            cv2.circle(self.feature_detection.input_image, pt1, 3, (0, 0, 0), -1)
            cv2.circle(self.feature_detection.input_image, pt2, 3, (255, 255, 255), -1)
            cv2.arrowedLine(
                self.feature_detection.input_image,
                pt1,
                pt2,
                color,
                thickness,
                tipLength=tipLength,
            )

    def pose_estimation(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        abs_scale: float
    ) -> int:
        """
        Estimates relative pose between two sets of keypoints.

        Parameters:
            points1 (np.ndarray): Keypoints from previous frame, shape (N, 2).
            points2 (np.ndarray): Keypoints from current frame, shape (N, 2).
            abs_scale (float): Absolute scale for translation (used in RGB VO).

        Returns:
            int: 0 if pose update successful, 1 if rotation change too large.
        """
        E, mask = cv2.findEssentialMat(
            points2,
            points1,
            focal=self.focal,
            pp=self.pp,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        # best_num_inliers = 0
        # for _E in np.split(E, len(E) / 3):
        #     n, R_temp, t_temp, _ = cv2.recoverPose(
        #         _E, points2, points1, focal=self.focal, pp=self.pp, mask=mask)
        #     if n > best_num_inliers:
        #         best_num_inliers = n
        #         R_diff, t = R_temp, t_temp
        #         # ret = (R, t[:, 0], mask.ravel() > 0)

        if self.args.vo_type == "rgb":
            points, R_diff, t, mask = cv2.recoverPose(
                E,
                points2[mask.ravel().astype(bool)],
                points1[mask.ravel().astype(bool)],
                focal=self.focal,
                pp=self.pp,
            )

        if self.args.vo_type == "rgbd":
            pts2d, pts3d = self.depth_estimation(points1, points2)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, self.K, None
            )
            R_3d, _ = cv2.Rodrigues(rvec)

            # Camera to world transformation
            R_rel = R_3d
            t_rel = tvec.reshape(3, 1)

            R_rel_inv = R_rel.T
            t_rel_inv = -R_rel_inv @ t_rel

            R_diff = R_rel_inv
            t = t_rel_inv

            abs_scale = 1

        angles_change = R.from_matrix(R_diff).as_euler("zyx", degrees=True)
        if np.any(np.abs(angles_change) > 15):
            return 1
        
        # Update global pose
        self.t_total = self.t_total + abs_scale * self.R_total.dot(t)
        self.R_total = self.R_total.dot(R_diff)

        return 0

    def depth_estimation(
        self,
        kp1: list[tuple[float, float]],
        kp2: list[tuple[float, float]],
        factor: float = 5000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates 2D-3D point correspondences for pose estimation using RGB-D data.

        Parameters:
            kp1 (list[tuple[float, float]]): Keypoints from the previous frame (with depth), shape (N, 2).
            kp2 (list[tuple[float, float]]): Corresponding keypoints from the current frame, shape (N, 2).
            factor (float): Depth scaling factor (e.g., 5000 for mm to meters conversion).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - pts2d: 2D points in the current frame, shape (M, 1, 2)
                - pts3d: Corresponding 3D points in the previous frame, shape (M, 1, 3)
        """
        pts2d = []
        pts3d = []
        for p1, p2 in zip(kp1, kp2):
            u, v = p1
            d = self.past_depth[int(v), int(u)] / factor

            if d == 0:
                continue

            x = (u - self.K[0, 2]) * d / self.K[0, 0]
            y = (v - self.K[1, 2]) * d / self.K[1, 1]

            pts3d.append([x, y, d])
            pts2d.append(p2)

        pts3d = np.array(pts3d, dtype=np.float32).reshape(-1, 1, 3)
        pts2d = np.array(pts2d, dtype=np.float32).reshape(-1, 1, 2)

        return pts2d, pts3d

    def compute_first_image(self, image: np.ndarray, depth_image: np.ndarray | None = None) -> None:
        """
        Initializes the visual odometry pipeline using the first input image (and optional depth image).

        Parameters:
            image (np.ndarray): The first RGB image frame.
            depth_image (np.ndarray | None): The corresponding depth map (used for RGB-D odometry).
        """        
        self.feature_detection.get_input(image)
        self.past_predictions["raw"], _, _ = self.feature_detection.process_Superpoint()
        self.past_predictions["descriptors"] = (
            self.past_predictions["raw"]["descriptors"][0]
            .cpu()
            .detach()
            .numpy()
            .astype(np.float32)
        )
        points_th = self.past_predictions["raw"]["keypoints"][0]
        keypoints_np = np.array(points_th)
        self.past_predictions["keypoints"] = [
            cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np
        ]

        if depth_image is not None and self.args.vo_type == "rgbd":
            self.past_depth = depth_image

        # gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        # self.past_predictions['keypoints'], self.past_predictions['descriptors'] = sift.detectAndCompute(gray,None)

    def compute_pipeline(
        self,
        image: np.ndarray,
        pos_cur: Optional[np.ndarray] = None,
        pos_prev: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
    ) -> Tuple[int, Tuple[float, float, float, float, float, int]]:
        """
        Runs the full visual odometry pipeline for a single frame.

        This method processes the input image through SuperPoint for keypoint detection,
        matches keypoints using either brute-force or SuperGlue, estimates camera pose,
        updates the trajectory and orientation, and optionally draws motion arrows.

        Args:
            image (np.ndarray): Current RGB image frame.
            pos_cur (np.ndarray, optional): Current ground truth position (used to compute scale). Defaults to None.
            pos_prev (np.ndarray, optional): Previous ground truth position. Defaults to None.
            depth_image (np.ndarray, optional): Depth image corresponding to the RGB frame (used in RGB-D mode). Defaults to None.

        Returns:
            Tuple[int, Tuple[float, float, float, float, float, int]]:
                - result (int): 0 if pose estimation succeeded, 1 if discarded due to large rotation.
                - timings (Tuple): Durations of each stage in seconds:
                    * preprocessing_time (float)
                    * superpoint_time (float)
                    * postprocessing_time (float)
                    * matching_time (float)
                    * pose_estimation_time (float)
                    * num_keypoints (int): Number of keypoints detected in current frame.
        """

        start = time.perf_counter()
        # preprocessing
        self.feature_detection.get_input(image)
        pre = time.perf_counter()
        # superPoint
        self.present_predictions["raw"], net, post = (
           self.feature_detection.process_Superpoint()
        )
        # net = 0
        # post = 0
        match = time.perf_counter()
        # matching
        self.present_predictions["descriptors"] = (
            self.present_predictions["raw"]["descriptors"][0]
            .cpu()
            .detach()
            .numpy()
            .astype(np.float32)
        )
        points_th = self.present_predictions["raw"]["keypoints"][0]
        keypoints_np = np.array(points_th)
        self.present_predictions["keypoints"] = [
            cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np
        ]
        # gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        # self.present_predictions['keypoints'], self.present_predictions['descriptors'] = sift.detectAndCompute(gray,None)

        if depth_image is not None and self.args.vo_type == "rgbd":
            self.present_depth = depth_image

        if self.matching_type == "bf":
            m_kp1, m_kp2 = self.match_descriptors_bf()
            H, inliers = self.compute_homography(m_kp1, m_kp2)
            self.matches = np.array(self.matches)[inliers.astype(bool)].tolist()
            pts1 = np.float32(
                [
                    self.past_predictions["keypoints"][match.queryIdx].pt
                    for match in self.matches
                ]
            )  # shape (N, 2)
            pts2 = np.float32(
                [
                    self.present_predictions["keypoints"][match.trainIdx].pt
                    for match in self.matches
                ]
            )  # shape (N, 2)
        if self.matching_type == "SuperGlue":
            sg_matching = self.superglue_match()
            sg_kpts0 = self.past_predictions["raw"]["keypoints"][0].cpu().numpy()
            sg_kpts1 = sg_matching["keypoints1"][0].cpu().numpy()
            sg_matches = sg_matching["matches0"][0].cpu().numpy()
            sg_confidence = sg_matching["matching_scores0"][0].detach().cpu().numpy()

            sg_valid = sg_matches > -1
            m_kp1 = sg_kpts0[sg_valid]
            m_kp2 = sg_kpts1[sg_matches[sg_valid]]
            H, inliers = self.compute_homography(m_kp1, m_kp2)
            pts1 = m_kp1[inliers.astype(bool)]
            pts2 = m_kp2[inliers.astype(bool)]

        end = time.perf_counter()
        # pose estimation
        if pos_cur is None:
            abs_scale = 1
        else:
            abs_scale = np.linalg.norm(pos_cur - pos_prev)
            # print("Scale: ", abs_scale)

        result = self.pose_estimation(pts1, pts2, abs_scale)

        r_global = self.start_R.dot(self.R_total)
        r = R.from_matrix(r_global)
        angles = r.as_euler("zyx", degrees=True)  # yaw, pitch, roll
        self.R_list.append(angles)
        # trac = self.t_total + self.start_t
        trac = self.start_R.dot(self.t_total) + self.start_t
        self.trajectory.append(trac.flatten().tolist())
        position_time = time.perf_counter()
        self.show_arrows(pts1, pts2)

        return result, (pre - start, net, post, end - match, position_time - end, len(self.present_predictions["keypoints"]))
