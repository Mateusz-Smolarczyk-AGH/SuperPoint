# /bin/python3

from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import processing
from evo.core import sync
from evo.core.metrics import PoseRelation, APE, RPE
from evo.core.trajectory import PoseTrajectory3D, PosePath3D
from evo.tools import file_interface, plot
from pathlib import Path
from evo.core.units import Unit

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def plot_trajectory(
    pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True
):
    assert isinstance(pred_traj, PoseTrajectory3D)

    if gt_traj is not None:
        assert isinstance(gt_traj, PoseTrajectory3D)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy  # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

def corect_trajectory(groundtruth, quaternions, timestamps, scene, trajectory):
    traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
    traj_est = PoseTrajectory3D(
            positions_xyz=trajectory[:, :3],
            orientations_quat_wxyz=quaternions[:, [3, 0, 1, 2]],
            timestamps=timestamps,
        )
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    Path("saved_trajectories").mkdir(exist_ok=True)
    file_interface.write_tum_trajectory_file(f"saved_trajectories/TUM_RGBD_{scene}_Trial{1:02d}_our_raw3.txt", traj_est)


    Path("trajectory_plots").mkdir(exist_ok=True)
    plot_trajectory(
        traj_est,
        traj_ref,
        f"TUM-RGBD Frieburg1 {scene} Trial (ATE: {0:.03f})",
        f"trajectory_plots/TUM_RGBD_Frieburg1_{scene}_Trial{1:02d}_our_raw3.pdf",
        align=True,
        correct_scale=False,
    )

def compute_sequence(image_folder, t_gt, start_r, matching_type, database='tum'):
    pre_times = []
    net_times = []
    post_times = []
    matching_times = []
    all_times = []
    matches_list = []

    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".ppm"))]
    )
    lenght = len(image_files)
    if database == 'tum':
        camera = processing.PinholeCamera(517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054,	0.0026,	1.1633)
    if database == "kitti":
        camera = processing.PinholeCamera(
            fx=707.0912,
            fy=707.0912,
            cx=601.8873,
            cy=183.1104,
            d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0  # brak danych o dystorsji w P
        )
    # camera = processing.PinholeCamera(520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172)
    K_l = np.array([camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0]).reshape(3, 3)

    start_t = t_gt[0]
    t_start = np.array([[start_t[0]],
                        [start_t[1]],
                        [start_t[2]]])

    # Rotacja z kwaternionu
    r = R.from_euler("zyx",start_r,degrees=True)
    start_R = r.as_matrix()

    first_image = cv2.imread(os.path.join(image_folder, image_files[start]))
    # image_size = (first_image.shape[1], first_image.shape[0])
    image_size = (416, 128)
    # first_image, offset = processing.crop_center(first_image, image_size)
    # camera.cx=camera.cx - offset[0]
    # camera.cy=camera.cy - offset[1]
    scale_x = image_size[0] / first_image.shape[1]
    scale_y = image_size[1] / first_image.shape[0]

    camera.fx = camera.fx * scale_x
    camera.fy = camera.fy * scale_y
    camera.cx = camera.cx * scale_x
    camera.cy = camera.cy * scale_y
    if database == "tum":
        first_image = cv2.resize(first_image, (image_size[0] + 32, image_size[1] + 16))
        first_image = cv2.undistort(first_image, K_l, camera.d) 
        first_image = first_image[8:-8, 16:-16, :]
    else:
        first_image = cv2.resize(first_image, image_size)

    odometry = processing.VisualOdometry(image_size, start_R, t_start, camera, matching_type, database)
    odometry.compute_first_image(first_image)
    for i in range(start+1, end):
        print(f"Processing: {(i-(start+1))/(end-start+1) * 100}%")
        image_file = image_files[i]
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        skip_frame, time = odometry.compute_pipeline(image, t_gt[i-start], t_gt[i-start - 1])
        if skip_frame == 0:
            odometry.past_predictions = odometry.present_predictions.copy()
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
    return t, gt_euler, time_array

def save_trajectory(t, euler_angles, time_array, output_file):
    # Upewnij się, że masz poprawne wymiary
    assert t.shape[0] == euler_angles.shape[0], "Długość wektorów t i euler_angles musi być taka sama"

    # Zamień eulery na kwaterniony
    rot = R.from_euler('zyx', euler_angles, degrees=True)
    quats = rot.as_quat()  # scipy: [x, y, z, w]

    # Zbuduj wynikową tablicę
    output_data = np.hstack((
        time_array.reshape(-1, 1),  # timestamp
        t,                         # pozycja: tx, ty, tz
        quats                      # orientacja: qx, qy, qz, qw
    ))

    # Zapisz do pliku
    np.savetxt(output_file, output_data, fmt="%.4f")

def normalized_ape(gt_coords, est_coords):
    gt = np.array(gt_coords)
    est = np.array(est_coords)

    ape = np.mean(np.linalg.norm(gt - est, axis=1))
    gt_diffs = np.linalg.norm(gt[1:] - gt[:-1], axis=1)
    traj_length = np.sum(gt_diffs)

    return ape / traj_length if traj_length > 0 else np.inf

def trajectory_rpe(gt_coords, est_coords, delta=1):
    gt = np.array(gt_coords)
    est = np.array(est_coords)
    N = len(gt) - delta
    errors = []
    for i in range(N):
        gt_diff = gt[i + delta] - gt[i]
        est_diff = est[i + delta] - est[i]
        denom = np.linalg.norm(gt_diff)
        if denom > 1e-8:
            error = np.linalg.norm(gt_diff - est_diff) / denom
        else:
            error = 0.0       
        errors.append(error)
    return np.mean(errors)

def angular_mae(gt_angles_deg, est_angles_deg):
    gt_angles = np.radians(gt_angles_deg)
    est_angles = np.radians(est_angles_deg)
    
    angle_diff = gt_angles - est_angles
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # wrap do [-π, π]

    mae_rad = np.mean(np.abs(angle_diff))
    mae_deg = np.degrees(mae_rad)
    return mae_deg

def plot_result(trajectory, gt_t, est_euler, gt_euler):    
    tx_est, ty_est, tz_est = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    tx, ty, tz = gt_t[:, 0], gt_t[:, 1], gt_t[:, 2]

    print("Średni błąd kąta:", angular_mae(gt_euler, est_euler), "stopni")
    print("RPE trajektorii:", trajectory_rpe(gt_t, trajectory))
    print(f"Norm APE trajektorii: {normalized_ape(gt_t, trajectory) * 100} %")

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
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    # wyśrodkuj osie
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
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

def save_trajectory_with_euler(filename, positions, euler_angles):
    """
    Zapisuje trajektorię i kąty Eulera do pliku tekstowego.
    
    :param filename: Ścieżka do pliku wyjściowego
    :param positions: ndarray (N, 3), pozycje w metrach
    :param euler_angles: ndarray (N, 3), kąty Eulera (rad lub deg)
    :param degrees: jeśli True, zapisuje kąty w stopniach
    """
    # Łączymy dane w jedną macierz: [x y z roll pitch yaw]
    data = np.hstack((positions, euler_angles))

    # Zapis do pliku z nagłówkiem
    header = "x y z roll pitch yaw"
    np.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    print(f"Zapisano do pliku: {filename}")

def metrics(trajectory, est_euler, gt):
    poses_se3 = []
    gt_cut = PosePath3D(poses_se3=gt.poses_se3[start:end])

    for pos, euler in zip(trajectory, est_euler):
        rot = R.from_euler('xyz', euler).as_matrix()  # 3x3 macierz rotacji
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        poses_se3.append(T)
    traj = PosePath3D(poses_se3=poses_se3)
    ape_metric = APE(PoseRelation.translation_part)
    ape_metric.process_data((gt_cut, traj))
    ape_result = ape_metric.get_result()
    print("ATE RMSE:", ape_result.stats["rmse"])

    rpe_metric = RPE(PoseRelation.translation_part, delta=1, delta_unit=Unit.frames)
    rpe_metric.process_data((gt_cut, traj))
    print("RPE RMSE:", rpe_metric.get_result().stats['rmse'])

def metrics_from_file(scene, database, start, end, comment):
    data = np.loadtxt(f"results//{database}_{scene}_{start}_{end}_{comment}.txt", skiprows=1)
    positions = data[:, 0:3]
    euler_angles = data[:, 3:6]
    groundtruth = "data//dataset//poses//" + f"{scene}.txt"
    pose = file_interface.read_kitti_poses_file(groundtruth)
    metrics(positions, euler_angles, pose)
    poses = pose.positions_xyz[start:end]
    length = 0
    for i in range(1, len(poses)):
        p1 = poses[i-1]  # pozycja xyz z macierzy 4x4
        p2 = poses[i]
        length += np.linalg.norm(p2 - p1)
    print(f"Total Lenght: {length}")
    print(f"Frames: {end-start}")

def main(scene, database, start, end, comment):
    if database == 'tum':       
        file = "data//" + scene
        gt_t, gt_euler, time_array = get_gt(start, end, file)
        file += r"\rgb"
    if database == 'kitti':
        groundtruth = "data//dataset//poses//" + f"{scene}.txt"
        pose = file_interface.read_kitti_poses_file(groundtruth)
        gt_t = pose.positions_xyz[start:end]
        quaternions = pose.orientations_quat_wxyz[start:end]
        quats_xyzw = quaternions[:, [1, 2, 3, 0]]

        gt_rot = R.from_quat(quats_xyzw)  # scipy używa kolejności: x, y, z, w
        gt_euler = gt_rot.as_euler('zyx', degrees=True)  # yaw, pitch, roll
        file = 'data//dataset//sequences//' + scene + "//image_0"
        
    trajectory, est_euler = compute_sequence(file, gt_t, gt_euler[0], "SuperGlue", database)

    if database == 'tum':
        quaternions = R.from_euler("zyx", est_euler, degrees=True).as_quat()
        groundtruth = file + r"\groundtruth.txt"
        corect_trajectory(groundtruth, quaternions, time_array, scene)
        save_trajectory(trajectory, est_euler, time_array, "results/generated_trajectory.txt")
    if database == "kitti":
        metrics(trajectory, est_euler, pose)
    save_trajectory_with_euler(f"results//{database}_{scene}_{start}_{end}_{comment}.txt", trajectory, est_euler)
    plot_result(trajectory, gt_t, est_euler, gt_euler)
# scene = "00"
# groundtruth = "data//dataset//poses//" + f"{scene}.txt"

# pose = file_interface.read_kitti_poses_file(groundtruth)



# print(length)
scene = "00"
database = 'kitti'
start = 0
end = 10
comment = "416x128_SuperPoint_SuperGlue"
# main(scene, database, start, end, comment)
metrics_from_file(scene, database, start, end, comment)