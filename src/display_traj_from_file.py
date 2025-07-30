# /bin/python3

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import processing
from evo.core import sync, metrics
from evo.core.metrics import PoseRelation, APE, RPE

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.trajectory import PoseTrajectory3D, PosePath3D
from evo.tools import file_interface, plot
from pathlib import Path
from evo.core.units import Unit

import evo.common_ape_rpe as common

import os
from pathlib import Path
import argparse


def get_gt(start, end, file):
    trajectory_gt_data = np.loadtxt(
        file / "groundtruth.txt"
    )  # lub zamiast pliku: zrób z listy stringów
    gt_timestamps = trajectory_gt_data[:, 0]

    # Wczytaj timestampy z pliku z obrazami (np. 2 kolumny: timestamp filename)
    timestamps_img = []
    with open(file / "rgb.txt") as f:
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
    t = matched_gt[:, 1:4]
    return t, gt_euler, time_array


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
    plot_mode = plot.PlotMode.xyz  # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")

        start_gt = gt_traj.positions_xyz[0]
        ax.scatter(
            start_gt[0],
            start_gt[1],
            start_gt[2],
            color="green",
            marker="x",
            s=50,
            label="Starting Point",
        )
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    # plt.close(fig=fig)
    print(f"Saved {filename}")


def corect_trajectory(groundtruth, quaternions, timestamps, scene, trajectory):
    traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
    traj_est = PoseTrajectory3D(
        positions_xyz=trajectory[:, :3],
        orientations_quat_wxyz=quaternions[:, [3, 0, 1, 2]],
        timestamps=timestamps,
    )
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result_ape = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.rotation_part,
        align=True,
        correct_scale=True,
    )
    ate_score = result_ape.stats["rmse"]

    # Oblicz RPE
    result_rpe = main_rpe.rpe(
        traj_ref,
        traj_est,
        pose_relation=PoseRelation.rotation_part,
        delta=1,  # Odstęp czasowy (np. 1 klatka)
        delta_unit=Unit.frames,
        align=True,
        correct_scale=True,
    )
    rpe_score = result_rpe.stats["rmse"]

    print(f"ATE: {ate_score:.04f} m")
    print(f"RPE: {rpe_score:.04f} m/s")
    


    Path("saved_trajectories").mkdir(exist_ok=True)
    file_interface.write_tum_trajectory_file(
        f"saved_trajectories/TUM_RGBD_{scene}_Trial{1:02d}_our_raw3.txt", traj_est
    )

    Path("trajectory_plots").mkdir(exist_ok=True)
    plot_trajectory(
        traj_est,
        traj_ref,
        "",  # TUM-RGBD Frieburg1 {scene} Trial (ATE: {0:.03f})
        f"trajectory_plots/TUM_RGBD_Frieburg1_{scene}_Trial{1:02d}_our_raw3.pdf",
        align=False,
        correct_scale=False,
    )


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

    # print("Średni błąd kąta:", angular_mae(gt_euler, est_euler), "stopni")
    # print("RPE trajektorii:", trajectory_rpe(gt_t, trajectory))
    # print(f"Norm APE trajektorii: {normalized_ape(gt_t, trajectory) * 100} %")

    # Wykres 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tx_est, ty_est, tz_est, label="Estimated", color="blue")
    ax.plot(tx, ty, tz, "--", label="GT", color="gray")

    ax.scatter(
        tx_est[0],
        ty_est[0],
        tz_est[0],
        color="green",
        marker="x",
        s=25,
        label="Starting point",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Trajektoria estymowana")
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

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.legend()

    plt.figure(figsize=(12, 6))

    labels = ["Yaw (deg)", "Pitch (deg)", "Roll (deg)"]
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(est_euler[:, i], label="Estimated")
        plt.plot(gt_euler[:, i], label="GT", linestyle="--")

        plt.ylabel(labels[i])
        plt.legend()
        plt.grid(True)

    plt.xlabel("Number of frames")
    # plt.suptitle("Porównanie kątów Eulera kamery (Estymacja vs GT)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ### Main configuration ###
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maindir",
        type=Path,
        default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint",
    )
    parser.add_argument(
        "--superglue_weights", type=str, default="weights/superglue_indoor.pth"
    )

    parser.add_argument("--vo_type", type=str, default="rgbd")  # [rgb, rgbd]
    parser.add_argument("--database", type=str, default="tum")  # [tum, kitti]
    # parser.add_argument("--network", type=str, default="/uczelnia/Repositorium/superpoint-fpga/pytorch-superpoint/logs/superpoint_coco_4bity_relu_douczenie/checkpoints/superPointNet_9200_checkpoint.pth.tar")
    # parser.add_argument("--network",type=str, default="/uczelnia/Repositorium/superpoint-fpga/pytorch-superpoint/logs/superpoint_coco_3bity/checkpoints/superPointNet_190000_checkpoint.pth.tar")
    # parser.add_argument("--network", type=str, default="/uczelnia/Repositorium/superpoint-fpga/pytorch-superpoint/logs/superpoint_coco/checkpoints/superPointNet_106000_checkpoint.pth.tar")
    # parser.add_argument("--network", type=str, default="/uczelnia/Repositorium/superpoint-fpga/pytorch-superpoint/logs/superpoint_coco/checkpoints/superPointNet_103200_checkpoint.pth.tar")
    # parser.add_argument("--network", type=str, default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint/weights/superpoint_v1.pth")

    parser.add_argument(
        "--kittidir",
        type=Path,
        default="/uczelnia/Repositorium/superpoint-fpga/data_odometry_color/dataset",
    )
    parser.add_argument(
        "--tumdir",
        type=Path,
        default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint/data",
    )

    parser.add_argument("--tum_seq", type=str, default="rgbd_dataset_freiburg1_floor")
    parser.add_argument("--kitti_seq", type=str, default="00")
    parser.add_argument("--kitti_gt", type=Path, default="datasets/KITTI/poses/00.txt")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1241)
    parser.add_argument("--viz", action="store_true", default=True)
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_trajectory", default=True, action="store_true")
    args = parser.parse_args()


# open the txt file - args.maindir / "results/generated_trajectory.txt"
# file_results = args.maindir / "results/tum_freiburg1_floor_8bit.txt"
file_results = args.maindir / "results/tum_freiburg1_floor_float.txt"
# file_results = args.maindir / "results/tum_freiburg1_floor_4_2_4bit.txt"
if not file_results.is_file():
    print(f"File {file_results} does not exist.")
    exit(1)

file = args.tumdir / args.tum_seq
gt_t, gt_euler, time_array = get_gt(args.start, args.end, file)


# Read the trajectory file
data = np.loadtxt(file_results, delimiter=" ")
print(data.shape)

est_timestamp = data[:, 0]
est_trajectory = data[:, 1:4]
est_quat = data[:, 4:8]

est_rot = R.from_quat(est_quat)  # scipy używa kolejności: x, y, z, w
est_euler = est_rot.as_euler("zyx", degrees=True)  # yaw, pitch, roll


# quaternions = R.from_euler("zyx", est_euler, degrees=True).as_quat()
groundtruth = file / "groundtruth.txt"

corect_trajectory(groundtruth, est_quat, est_timestamp, args.tum_seq, est_trajectory)


# save_trajectory(
#     trajectory,
#     est_euler,
#     est_timestamp,
#     args.maindir / "results/generated_trajectory.txt",
# )


plot_result(est_trajectory, gt_t, est_euler, gt_euler)
