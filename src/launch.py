# /bin/python3

import time
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import superpoint_pytorch
import processing
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
start = 0
end = 15

def compute_sequence(data_folder, t_gt, start_r):
    pre_times = []
    net_times = []
    post_times = []
    matching_times = []
    all_times = []
    matches_list = []
    matches = []

    image_folder = data_folder + r"\rgb"
    trajectory_gt = np.loadtxt(data_folder + "\groundtruth.txt")

    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".ppm"))]
    )
    lenght = len(image_files)

    camera = processing.PinholeCamera(517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054,	0.0026,	1.1633)
    start_t = t_gt[0]
    t_start = np.array([[start_t[0]],
                        [start_t[1]],
                        [start_t[2]]])

    # Rotacja z kwaternionu
    r = R.from_euler("zyx",start_r,degrees=True)
    start_R = r.as_matrix()

    first_image = cv2.imread(os.path.join(image_folder, image_files[start]))
    image_size = (first_image.shape[1], first_image.shape[0])
    odometry = processing.VisualOdometry(image_size, start_R, t_start, camera)
    odometry.compute_first_image(first_image)
    for i in range(start+1, end):
        print(f"Processing: {(i-(start+1))/(end-start+1) * 100}%")
        image_file = image_files[i]
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        skip_frame, time = odometry.compute_pipeline(image, t_gt[i-start], t_gt[i-start - 1])
        if skip_frame == 0:
            odometry.keypoints["past"] = odometry.keypoints["present"]
            odometry.descriptors["past"] = odometry.descriptors["present"]
        cv2.imshow("Film", odometry.feature_detection.input_image)

        # Czekaj 30 ms na kolejny obraz (około 30 FPS)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        pre_times.append(time[0])
        net_times.append(time[1])
        post_times.append(time[2])
        matching_times.append(time[3])
        matches_list.append(time[4])
    num_iterations = len(pre_times) - 1
    avg_pre = sum(pre_times[1:]) / num_iterations * 1000
    avg_net = sum(net_times[1:]) / num_iterations * 1000
    avg_post = sum(post_times[1:]) / num_iterations * 1000
    avg_matching = sum(matching_times[1:]) / num_iterations * 1000
    avg_all = sum(all_times[1:]) / num_iterations * 1000
    avg_matches = sum(matches_list[1:]) / num_iterations

    print(f"Średnie czasy (ms): pre: {avg_pre:.6f} | net: {avg_net:.6f} | post: {avg_post:.6f} | matching: {avg_matching:.6f} | all: {avg_all:.6f} | matches: {avg_matches:.6f}")
    return np.array(odometry.trajectory), np.array(odometry.R_list)

def get_gt(start, end, file):
    trajectory_gt_data = np.loadtxt(file + r"\groundtruth.txt")  # lub zamiast pliku: zrób z listy stringów
    gt_timestamps = trajectory_gt_data[:, 0]

    # Wczytaj timestampy z pliku z obrazami (np. 2 kolumny: timestamp filename)
    timestamps_img = []
    with open(file + r"\rgb.txt") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            ts = float(line.strip().split()[0])
            timestamps_img.append(ts)

    timestamps_img = np.array(timestamps_img)
    time_array = timestamps_img[start:end]
    closest_gt_indices = np.abs(gt_timestamps[:, np.newaxis] - time_array).argmin(axis=0)
    matched_gt = trajectory_gt_data[closest_gt_indices]

    quaternions = matched_gt[:, 4:8]  # qx, qy, qz, qw
    gt_rot = R.from_quat(quaternions)  # scipy używa kolejności: x, y, z, w
    gt_euler = gt_rot.as_euler('zyx', degrees=True)  # yaw, pitch, roll
    t = matched_gt[:, 1:4]
    return t, gt_euler

file = r"data\rgbd_dataset_freiburg1_xyz"
gt_t, gt_euler = get_gt(start, end, file)
trajectory, est_euler = compute_sequence(file, gt_t, gt_euler[0])
tx_est, ty_est, tz_est = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
tx, ty, tz = gt_t[:, 0], gt_t[:, 1], gt_t[:, 2]
# Wykres 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(tx_est, ty_est, tz_est, label='Trajektoria estymowana', color='red')
ax.plot(tx, ty, tz, label='Trajektoria GT', color='blue')

ax.scatter(tx_est[0], ty_est[0], tz_est[0], color='black', marker='o', s=50, label='Punkt startowy')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Trajektoria estymowana")
ax.legend()

plt.figure(figsize=(12, 6))

labels = ['Yaw (Z)', 'Pitch (Y)', 'Roll (X)']
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(est_euler[:, i], label='Estymowane')
    plt.plot(gt_euler[:, i], label='Ground Truth', linestyle='--')

    plt.ylabel(labels[i])
    plt.legend()
    plt.grid(True)

plt.xlabel('Krok czasowy')
plt.suptitle("Porównanie kątów Eulera kamery (Estymacja vs GT)")
plt.tight_layout()
plt.show()