import time

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import superpoint_pytorch
import processing
import os
from tqdm import tqdm
import glob
from pathlib import Path

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
start = 30
end = 200


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
    plot_mode = plot.PlotMode.xz  # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")


def compute_sequence(data_folder, start_t, start_r, weight_path):
    pre_times = []
    net_times = []
    post_times = []
    matching_times = []
    all_times = []
    matches_list = []
    matches = []

    # image_folder = data_folder + r"\rgb"
    image_folder = os.path.join(data_folder, "rgb")
    # trajectory_gt = np.loadtxt(data_folder + "\groundtruth.txt")
    # trajectory_gt = os.path.join(data_folder, "groundtruth.txt")

    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".ppm"))]
    )

    camera_matrix = processing.PinholeCamera(
        517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633
    )
    t_start = np.array([[start_t[0]], [start_t[1]], [start_t[2]]])

    # Rotacja z kwaternionu
    r = R.from_euler("zyx", start_r, degrees=True)
    start_R = r.as_matrix()

    first_image = cv2.imread(os.path.join(image_folder, image_files[start]))
    image_size = (first_image.shape[1], first_image.shape[0])
    odometry = processing.VisualOdometry(
        image_size, start_R, t_start, camera_matrix, weight_path
    )
    odometry.compute_first_image(first_image)
    for i in tqdm(range(start + 1, end)):
        # print(f"Processing: {(i-(start+1))/(end-start+1) * 100}%")
        image_file = image_files[i]
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        skip_frame, time = odometry.compute_pipeline(image)
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

    print(
        f"Średnie czasy (ms): pre: {avg_pre:.6f} | net: {avg_net:.6f} | post: {avg_post:.6f} | matching: {avg_matching:.6f} | all: {avg_all:.6f} | matches: {avg_matches:.6f}"
    )
    return np.array(odometry.trajectory), np.array(odometry.R_list)


def get_gt(start, end, file):
    # trajectory_gt_data = np.loadtxt(file + r"\groundtruth.txt")  # lub zamiast pliku: zrób z listy stringów
    trajectory_gt_data = np.loadtxt(os.path.join(file, "groundtruth.txt"))
    gt_timestamps = trajectory_gt_data[:, 0]

    # Wczytaj timestampy z pliku z obrazami (np. 2 kolumny: timestamp filename)
    timestamps_img = []

    with open(os.path.join(file, "rgb.txt")) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            ts = float(line.strip().split()[0])
            timestamps_img.append(ts)

    timestamps_img = np.array(timestamps_img)
    time_array = timestamps_img[start:end]
    closest_gt_indices = np.abs(gt_timestamps[:, np.newaxis] - time_array).argmin(
        axis=0
    )
    matched_gt = trajectory_gt_data[closest_gt_indices]

    quaternions = matched_gt[:, 4:8]  # qx, qy, qz, qw
    gt_rot = R.from_quat(quaternions)  # scipy używa kolejności: x, y, z, w
    gt_euler = gt_rot.as_euler("zyx", degrees=True)  # yaw, pitch, roll
    tx, ty, tz = matched_gt[:, 1], matched_gt[:, 2], matched_gt[:, 3]
    return tx, ty, tz, gt_euler, timestamps_img


if __name__ == "__main__":

    ### Main configuration ###
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maindir",
        type=Path,
        default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint",
    )
    parser.add_argument("--network", type=str, default="weights/superpoint_v1.pth")
    parser.add_argument("--kittidir", type=Path, default="datasets/KITTI")
    parser.add_argument("--kitti_seq", type=str, default="00")

    parser.add_argument("--kitti_gt", type=Path, default="datasets/KITTI/poses/00.txt")

    parser.add_argument(
        "--tumdir",
        type=Path,
        default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint/data",
    )
    parser.add_argument("--tum_seq", type=str, default="rgbd_dataset_freiburg1_floor")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)

    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--show_img", action="store_true")

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # tum_scenes = [
    # "rgbd_dataset_freiburg1_360",
    # "rgbd_dataset_freiburg1_desk",
    # "rgbd_dataset_freiburg1_desk2",
    # "rgbd_dataset_freiburg1_floor",
    # "rgbd_dataset_freiburg1_plant",
    # "rgbd_dataset_freiburg1_room",
    # "rgbd_dataset_freiburg1_rpy",
    # "rgbd_dataset_freiburg1_teddy",
    # "rgbd_dataset_freiburg1_xyz",
    # ]
    
    superpoint_weights = args.maindir / args.network

    scene_dir = args.tumdir / args.tum_seq
    groundtruth = scene_dir / "groundtruth.txt"

    traj_ref = file_interface.read_tum_trajectory_file(groundtruth)

    images_dir = scene_dir / "rgb"

    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3, 3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    image_list = sorted(images_dir.glob("*.png"))

    pre_times = []
    net_times = []
    post_times = []
    matching_times = []
    all_times = []
    matches_list = []
    matches = []

    camera_matrix = processing.PinholeCamera(
        517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633
    )
    start_t = (0, 0, 0)  # translation
    start_r = (0, 0, 0)  # rotation
    t_start = np.array([[start_t[0]], [start_t[1]], [start_t[2]]])

    # Rotacja z kwaternionu
    r = R.from_euler("zyx", start_r, degrees=True)
    start_R = r.as_matrix()

    first_image = cv2.imread(str(image_list[0]))
    first_image = cv2.undistort(first_image, K_l, d_l)
    first_image = first_image[8:-8, 16:-16, :]

    image_size = (first_image.shape[1], first_image.shape[0])
    odometry = processing.VisualOdometry(
        image_size, start_R, t_start, camera_matrix, superpoint_weights=superpoint_weights
    )
    odometry.compute_first_image(first_image)
    timestamps = []

    il = 0
    for imfile in tqdm(image_list):

        if imfile == image_list[0]:
            continue

        image = cv2.imread(str(imfile))
        image = cv2.undistort(image, K_l, d_l)
        # image = image.transpose(2,0,1)

        intrinsics = np.asarray([fx, fy, cx, cy])

        # crop image to remove distortion boundary
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        # intrinsics = intrinsics[None]
        image = image[8:-8, 16:-16, :]

        timestamp = float(imfile.stem)
        timestamps.append(timestamp)

        if timestamp < 0:
            break

        # input_image = torch.as_tensor(image, device='cuda')

        skip_frame, time = odometry.compute_pipeline(image)
        if skip_frame == 0:
            odometry.keypoints["past"] = odometry.keypoints["present"]
            odometry.descriptors["past"] = odometry.descriptors["present"]

        # cv2.imshow("Film", odometry.feature_detection.input_image)

        # Czekaj 30 ms na kolejny obraz (około 30 FPS)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    trajectory = np.array(odometry.trajectory)
    est_euler = np.array(odometry.R_list)

    # convert to quaternion
    quaternions = R.from_euler("zyx", est_euler, degrees=True).as_quat()
    # convert to w, x, y, z

    traj_est = PoseTrajectory3D(
        positions_xyz=trajectory[:, :3],
        orientations_quat_wxyz=quaternions[:, [3, 0, 1, 2]],
        timestamps=timestamps,
    )

    tx_est, ty_est, tz_est = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)


    if args.save_trajectory:
        # Save trajectory to a text file
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(
            f"saved_trajectories/TUM_RGBD_{args.tum_seq}_Trial{1:02d}_our_raw3.txt", traj_est
        )

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(
            traj_est,
            traj_ref,
            f"TUM-RGBD Frieburg1 {args.scene} Trial (ATE: {0:.03f})",
            f"trajectory_plots/TUM_RGBD_Frieburg1_{args.scene}_Trial{1:02d}_our_raw3.pdf",
            align=False,
            correct_scale=False,
        )

    # # Save trajectory to a text file
    # output_file = os.path.join(main_dir, "estimated_trajectory.txt")
    # with open(output_file, "w") as f:
    #     for x, y, z in zip(tx_est, ty_est, tz_est):
    #         f.write(f"{x} {y} {z}\n")
    # print(f"Estimated trajectory saved to {output_file}")
    # # Wykres 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(tx_est, ty_est, tz_est, label="Trajektoria estymowana", color="red")
    # ax.scatter(
    #     tx_est[0],
    #     ty_est[0],
    #     tz_est[0],
    #     color="black",
    #     marker="o",
    #     s=50,
    #     label="Punkt startowy",
    # )
    # ax.set_xlim([-25, 25])
    # ax.set_ylim([-25, 25])
    # ax.set_zlim([-25, 25])
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Trajektoria estymowana")
    # ax.legend()

    # # Wykres 3D
    # fig1 = plt.figure()
    # ax2 = fig1.add_subplot(111, projection="3d")
    # ax2.plot(tx, ty, tz, label="Trajektoria GT", color="blue")
    # ax2.scatter(
    #     tx[0], ty[0], tz[0], color="black", marker="o", s=50, label="Punkt startowy"
    # )
    # ax2.set_xlim([1, 1.4])
    # ax2.set_ylim([-1.2, -0.8])
    # ax2.set_zlim([0.4, 0.8])
    # ax2.set_xlabel("X")
    # ax2.set_ylabel("Y")
    # ax2.set_zlabel("Z")
    # ax2.set_title("Trajektoria GT")
    # ax2.legend()

    # plt.figure(figsize=(12, 6))

    # labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    # for i in range(3):
    #     plt.subplot(3, 1, i + 1)
    #     plt.plot(est_euler[:, i], label="Estymowane")
    #     # plt.plot(gt_euler[:, i], label="Ground Truth", linestyle="--")

    #     plt.ylabel(labels[i])
    #     plt.legend()
    #     plt.grid(True)

    # plt.xlabel("Krok czasowy")
    # plt.suptitle("Porównanie kątów Eulera kamery (Estymacja vs GT)")
    # plt.tight_layout()
    # plt.show()


# def write_tum_trajectory_file(file_path: PathStrHandle, traj: PoseTrajectory3D,
#                               confirm_overwrite: bool = False) -> None:
#     """
#     :param file_path: desired text file for trajectory (string or handle)
#     :param traj: trajectory.PoseTrajectory3D
#     :param confirm_overwrite: whether to require user interaction
#            to overwrite existing files
#     """
#     if confirm_overwrite and isinstance(file_path, (str, Path)):
#         if not user.check_and_confirm_overwrite(file_path):
#             return
#     if not isinstance(traj, PoseTrajectory3D):
#         raise FileInterfaceException(
#             "trajectory must be a PoseTrajectory3D object")
#     stamps = traj.timestamps
#     xyz = traj.positions_xyz
#     # shift -1 column -> w in back column
#     quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
#     mat = np.column_stack((stamps, xyz, quat))
#     np.savetxt(file_path, mat, delimiter=" ")
#     if isinstance(file_path, str):
#         logger.info("Trajectory saved to: " + file_path)
