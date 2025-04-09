import time
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(torch.__version__)

detection_thresh = 0.005
nms_radius = 5

import superpoint_pytorch
import processing

def show_comparison(image1_path, image2_path, model, nettype='Normal'):
    img_size = (200, 200)
    image1, img1_orig = processing.preprocess_image(image1_path, img_size)
    image2, img2_orig = processing.preprocess_image(image2_path, img_size)

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
                pred_th_1 = processing.post_processing_short(scores, descriptors_dense, model.conf)
        # Extract descriptors
        descriptors = pred_th_1['descriptors'][0]
        points_th = pred_th_1['keypoints'][0]
        keypoints_np = np.array(points_th)  # Konwersja do NumPy
        keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        keypoints_list.append(keypoints)
        desc_list.append(descriptors.cpu().detach().numpy().astype(np.float32)
)

    m_kp1, m_kp2, matches = processing.match_descriptors(keypoints_list[0], desc_list[0], keypoints_list[1], desc_list[1])
    H, inliers = processing.compute_homography(m_kp1, m_kp2)

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

def sequence(image_folder):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".ppm"))])
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Wyświetlanie obrazów jako filmu
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            continue
        
        cv2.imshow("Film", frame)
        
        # Czekaj 30 ms na kolejny obraz (około 30 FPS)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break  # Wyjście z pętli po naciśnięciu 'q'

    cv2.destroyAllWindows()

    
def compute_sequence(image_folder, model, tryb='show'):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".ppm"))])
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape
    past_descriptors = None
    past_keypoints = None
    pre_times = []
    net_times = []
    post_times = []
    matching_times = []
    all_times = []
    matches_list = []
    matches = []
    R_list = []
    fx = 517.3  # focal length x
    fy = 516.5  # focal length y
    cx = 318.6  # optical center x
    cy = 255.3 # optical center y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Translacja
    t_total = np.array([[1.3563],
                        [0.6305],
                        [1.6380]])

    # Rotacja z kwaternionu
    q = [0.6132, 0.5962, -0.3311, -0.3986] 
    R_total = R.from_quat(q).as_matrix()
    trajectory = [t_total.flatten().tolist()]
    R_list = [R_total.tolist()]
    lenght = len(image_files)
    for i in range(0, 20):
        print(f"Processing: {i/20 * 100}%")
        image_file = image_files[i]
        image_path = os.path.join(image_folder, image_file)
        time1 = time.perf_counter()
        image, img_orig = processing.preprocess_image(image_path, (width, height))


        time2 = time.perf_counter()
        scores, descriptors_dense = model(torch.from_numpy(image[None, None]).float())
        time3 = time.perf_counter()
        pred_th_1 = processing.post_processing_short(scores, descriptors_dense, model.conf)
        descriptors = pred_th_1['descriptors'][0].cpu().detach().numpy().astype(np.float32)
        points_th = pred_th_1['keypoints'][0]
        keypoints_np = np.array(points_th)  # Konwersja do NumPy
        time4 = time.perf_counter()
        keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints_np]
        if past_descriptors is not None:
            m_kp1, m_kp2, matches = processing.match_descriptors(past_keypoints, past_descriptors, keypoints, descriptors)
            H, inliers = processing.compute_homography(m_kp1, m_kp2)
            matches = np.array(matches)[inliers.astype(bool)].tolist()

            if tryb == 'show':
                for match in matches:
                    pt1 = tuple(map(int, past_keypoints[match.queryIdx].pt))
                    pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
                    cv2.arrowedLine(img_orig, pt2, pt1, (0, 255, 0), 1, tipLength=0.2)

            if tryb == "points":
                points = np.array([[past_keypoints[match.queryIdx].pt, keypoints[match.trainIdx].pt] for match in matches], dtype=np.float32)
                m_kp1_float = points[:, 0]
                m_kp2_float = points[:, 1]
                t_total, R_total = pose_estimation(m_kp1_float, m_kp2_float, K, t_total, R_total)
                trajectory.append(t_total.flatten().tolist())
                R_list.append(R_total.tolist())

        past_descriptors = descriptors
        past_keypoints = keypoints
        end = time.perf_counter()

        if tryb=='show':
            cv2.imshow("Film", img_orig)
            
            # Czekaj 30 ms na kolejny obraz (około 30 FPS)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break  # Wyjście z pętli po naciśnięciu 'q'
        if tryb=='save':
            cv2.imwrite("data/matched/" + image_file, img_orig)
        if tryb=='time':
            pre_times.append(time2 - time1)
            net_times.append(time3 - time2)
            post_times.append(time4 - time3)
            matching_times.append(end - time4)
            all_times.append(end-time1)
            matches_list.append(len(matches))

    if tryb=='show':
        cv2.destroyAllWindows()
    if tryb=='time':
        num_iterations = len(pre_times) - 1
        avg_pre = sum(pre_times[1:]) / num_iterations * 1000
        avg_net = sum(net_times[1:]) / num_iterations * 1000
        avg_post = sum(post_times[1:]) / num_iterations * 1000
        avg_matching = sum(matching_times[1:]) / num_iterations * 1000
        avg_all = sum(all_times[1:]) / num_iterations * 1000
        avg_matches = sum(matches_list[1:]) / num_iterations
        return avg_pre, avg_net, avg_post, avg_matching, avg_all, 
    if tryb == 'points':
        return np.array(trajectory), np.array(R_list)
        
def pose_estimation(points1, points2, K, t_total, R_total):
    E, mask = cv2.findEssentialMat(points1, points2, K,  method=cv2.RANSAC, prob=0.999, threshold=3.0)
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    R_total = R_total @ R
    t_total = t_total + R_total @ t
    return t_total, R_total


model = superpoint_pytorch.SuperPoint_short(detection_threshold=detection_thresh, nms_radius=nms_radius).eval()
model.load_state_dict(torch.load("weights/superpoint_v6_from_tf.pth"))
image_folder = "data//rgbd_dataset_freiburg1_xyz//rgb"
trajectory, rotations_est = compute_sequence(image_folder, model, tryb='points')
# # print(f"Średnie czasy (ms): pre: {avg_pre:.6f} | net: {avg_net:.6f} | post: {avg_post:.6f} | matching: {avg_matching:.6f} | all: {avg_all:.6f} | matches: {avg_matches:.6f}")
trajectorygt = np.loadtxt(r"data\rgbd_dataset_freiburg1_xyz\groundtruth.txt")[:20]  # lub zamiast pliku: zrób z listy stringów


import matplotlib.pyplot as plt

# Wczytanie danych

# Rozdziel translacje
tx, ty, tz = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
txgt, tygt, tzgt = trajectorygt[:, 1], trajectorygt[:, 2], trajectorygt[:, 3]
# Wykres 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(tx, ty, tz, label='Trajektoria kamery', color='blue')
ax.plot(txgt, tygt, tzgt, label='Trajektoria kamery gt', color='red')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Trajektoria kamery 3D")
ax.legend()
plt.show()

def rotation_angles_from_matrices(rotation_list):
    return [np.degrees(R.from_matrix(Ri).magnitude()) for Ri in rotation_list]

# Konwersja kwaternionów do kątów (jeśli masz GT w postaci kwaternionów)
def rotation_angles_from_quaternions(quat_list):
    return [np.degrees(R.from_quat(q).magnitude()) for q in quat_list]

# Przykład: jeśli GT jest w kwaternionach
angles_est = rotation_angles_from_matrices(rotations_est)
angles_gt = rotation_angles_from_quaternions(trajectorygt[:, 4:])

# Wyrównaj długość (jeśli GT ma więcej klatek)
min_len = min(len(angles_est), len(angles_gt))
angles_est = angles_est[:min_len]
angles_gt = angles_gt[:min_len]

# Wykres
x = range(min_len)

plt.plot(x, angles_est, label="Odometria estymowana", color="blue")
plt.plot(x, angles_gt, label="Ground Truth", color="green", linestyle="--")
plt.xlabel("Klatka")
plt.ylabel("Kąt obrotu (°)")
plt.title("Porównanie kąta obrotu kamery")
plt.legend()
plt.grid()
plt.show()