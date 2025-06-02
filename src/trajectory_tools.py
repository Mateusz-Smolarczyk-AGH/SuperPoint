from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from evo.core import sync
from evo.core.metrics import PoseRelation, APE, RPE
from evo.core.trajectory import PoseTrajectory3D, PosePath3D
from evo.tools import file_interface, plot
from pathlib import Path
from evo.core.units import Unit
from pathlib import Path


def plot_trajectory(
    pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True
):
    """
    Plots a predicted trajectory (and optionally ground truth) in 2D and saves it as an image.

    This function can align and scale the predicted trajectory to the ground truth,
    visualize both on a 2D plot (XY plane), and save the resulting figure to a file.

    Parameters:
        pred_traj (PoseTrajectory3D): Predicted camera trajectory.
        gt_traj (PoseTrajectory3D, optional): Ground truth trajectory to compare with. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "".
        filename (str, optional): Output path to save the figure (e.g., "trajectory.png"). Defaults to "".
        align (bool, optional): Whether to align the predicted trajectory to the ground truth. Defaults to True.
        correct_scale (bool, optional): Whether to scale the predicted trajectory to match ground truth. Defaults to True.

    Raises:
        AssertionError: If `pred_traj` or `gt_traj` (if provided) are not instances of `PoseTrajectory3D`.

    Returns:
        None
    """
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


def corect_trajectory(
    groundtruth: str,
    quaternions: np.ndarray,      # shape: (N, 4), format: [x, y, z, w]
    timestamps: Union[np.ndarray, list],  # shape: (N,), float64 lub float32
    scene: str,
    trajectory: np.ndarray        # shape: (N, 3) lub (N, >=3)
    ):   
    """
    Aligns and evaluates an estimated trajectory against ground truth, saves both to file and plots them.

    This function loads the ground truth trajectory, builds the estimated trajectory from provided positions 
    and orientations, aligns them in time, saves the estimated trajectory in TUM format, and generates a 2D XY plot.

    Parameters:
        groundtruth (str): Path to the ground truth trajectory file in TUM format.
        quaternions (np.ndarray): Estimated orientations as Nx4 array in [x, y, z, w] format.
        timestamps (np.ndarray): Timestamps for the estimated poses, shape (N,).
        scene (str): Scene name used in the output filename.
        trajectory (np.ndarray): Estimated trajectory positions as Nx3 array [x, y, z].

    Returns:
        None

    Side effects:
        - Saves estimated trajectory to `saved_trajectories/TUM_RGBD_{scene}_Trial01_our_raw3.txt`.
        - Saves trajectory plot to `trajectory_plots/TUM_RGBD_Frieburg1_{scene}_Trial01_our_raw3.pdf`.
    """ 
    traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
    traj_est = PoseTrajectory3D(
        positions_xyz=trajectory[:, :3],
        orientations_quat_wxyz=quaternions[:, [3, 0, 1, 2]],
        timestamps=timestamps,
    )
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    Path("saved_trajectories").mkdir(exist_ok=True)
    file_interface.write_tum_trajectory_file(
        f"saved_trajectories/TUM_RGBD_{scene}_Trial{1:02d}_our_raw3.txt", traj_est
    )

    Path("trajectory_plots").mkdir(exist_ok=True)
    plot_trajectory(
        traj_est,
        traj_ref,
        f"TUM-RGBD Frieburg1 {scene} Trial (ATE: {0:.03f})",
        f"trajectory_plots/TUM_RGBD_Frieburg1_{scene}_Trial{1:02d}_our_raw3.pdf",
        align=True,
        correct_scale=False,
    )


def get_gt_tum(
    start: int,
    end: int,
    file: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    """
    Loads and aligns ground truth trajectory data with RGB image timestamps for TUM dataset.

    Parameters
    start : int
        Starting index of the image sequence.
    end : int
        Ending index of the image sequence.
    file : Path
        Path to the dataset folder containing 'groundtruth.txt' and 'rgb.txt'.

    Returns
    t : np.ndarray of shape (N, 3)
        Ground truth translation vectors [x, y, z] for matched timestamps.
    gt_euler : np.ndarray of shape (N, 3)
        Ground truth orientation in Euler angles [yaw, pitch, roll] in degrees.
    time_array : np.ndarray of shape (N,)
        Selected image timestamps that correspond to the matched ground truth poses.    
    """

    trajectory_gt_data = np.loadtxt(
        file / "groundtruth.txt"
    )
    gt_timestamps = trajectory_gt_data[:, 0]

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
    gt_rot = R.from_quat(quaternions)  # scipy uses: x, y, z, w
    gt_euler = gt_rot.as_euler("zyx", degrees=True)  # yaw, pitch, roll
    t = matched_gt[:, 1:4]
    return t, gt_euler, time_array


def save_trajectory(t: np.ndarray, euler_angles: np.ndarray, time_array: np.ndarray, output_file: str):
    """
    Saves trajectory data to a TUM-style text file.

    Parameters
    t : np.ndarray
        Translation vectors [x, y, z] for each timestamp.
    euler_angles : np.ndarray
        Orientation in Euler angles [yaw, pitch, roll] (degrees).
    time_array : np.ndarray
        Corresponding timestamps.
    output_file : str
        Path to the output file.
    """
    assert (
        t.shape[0] == euler_angles.shape[0]
    ), "The length of vectors t and euler_angles must be the same."

    rot = R.from_euler("zyx", euler_angles, degrees=True)
    quats = rot.as_quat()  # scipy: [x, y, z, w]

    output_data = np.hstack(
        (
            time_array.reshape(-1, 1),  # timestamp
            t,  # position: tx, ty, tz
            quats,  # orientation: qx, qy, qz, qw
        )
    )

    np.savetxt(output_file, output_data, fmt="%.4f")

def load_trajectory(input_file: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a trajectory file in TUM format and converts quaternions to Euler angles.

    Parameters
    input_file : str | pathlib.Path
        Path to the trajectory text file, expected format:
        ``timestamp tx ty tz qx qy qz qw``.

    Returns
    t : np.ndarray
        Translation vectors `[x, y, z]`, shape ``(N, 3)``.
    euler_angles : np.ndarray
        Orientation in Euler angles `[yaw, pitch, roll]` **in degrees**, shape ``(N, 3)``.
    time_array : np.ndarray
        Timestamps corresponding to each pose, shape ``(N,)``.
    """    
    data = np.loadtxt(input_file)

    time_array = data[:, 0]
    t = data[:, 1:4]               # tx, ty, tz
    quats = data[:, 4:8]           # qx, qy, qz, qw

    rot = R.from_quat(quats)
    euler_angles = rot.as_euler("zyx", degrees=True)

    return t, euler_angles, time_array


def plot_result(
    trajectory: np.ndarray,      # shape (N, 3), dtype=float
    gt_t: np.ndarray,            # shape (N, 3), dtype=float
    est_euler: np.ndarray,       # shape (N, 3), dtype=float
    gt_euler: np.ndarray         # shape (N, 3), dtype=float
):  
    """
    Visualizes estimated and ground truth camera trajectories and Euler angles.

    :param trajectory: Estimated camera positions, shape (N, 3), format [x, y, z].
    :param gt_t: Ground truth camera positions, shape (N, 3), format [x, y, z].
    :param est_euler: Estimated Euler angles in degrees, shape (N, 3), format [yaw, pitch, roll].
    :param gt_euler: Ground truth Euler angles in degrees, shape (N, 3), format [yaw, pitch, roll].
    :return: None. Shows plots using matplotlib.
    """
    tx_est, ty_est, tz_est = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    tx, ty, tz = gt_t[:, 0], gt_t[:, 1], gt_t[:, 2]

    # Wykres 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tx_est, ty_est, tz_est, label="Estimated trajectory", color="red")
    ax.plot(tx, ty, tz, label="GT trajectory", color="blue")

    ax.scatter(
        tx_est[0],
        ty_est[0],
        tz_est[0],
        color="black",
        marker="o",
        s=50,
        label="Start point",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Estimated trajectory")
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.legend()

    plt.figure(figsize=(12, 6))

    labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(est_euler[:, i], label="Estimated")
        plt.plot(gt_euler[:, i], label="Ground Truth", linestyle="--")

        plt.ylabel(labels[i])
        plt.legend()
        plt.grid(True)

    plt.xlabel("Time")
    plt.suptitle("Euler angles (Estymacja vs GT)")
    plt.tight_layout()
    plt.show()


def save_trajectory_with_euler(
    filename: str,
    positions: np.ndarray,      # shape (N, 3), position coordinates (x, y, z)
    euler_angles: np.ndarray    # shape (N, 3), Euler angles (roll, pitch, yaw) in degrees or radians
) -> None:
    """
    Saves trajectory positions and Euler angles to a text file.

    :param filename: Path to the output file
    :param positions: numpy.ndarray of shape (N, 3), positions in meters
    :param euler_angles: numpy.ndarray of shape (N, 3), Euler angles (roll, pitch, yaw) in radians or degrees
    :return: None
    """
    data = np.hstack((positions, euler_angles))

    header = "x y z roll pitch yaw"
    np.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments="")

    print(f"saved to file: {filename}")


def metrics(
    trajectory: np.ndarray,       # shape: (N, 3), estimated positions
    est_euler: np.ndarray,        # shape: (N, 3), estimated Euler angles [yaw, pitch, roll] in degrees
    gt: PosePath3D,               # full ground truth trajectory as a PosePath3D object
    start: int,                   # start index (inclusive)
    end: int                      # end index (exclusive)
    ) -> None: 
    """
    Computes and prints Absolute Pose Error (APE) and Relative Pose Error (RPE) metrics
    comparing estimated trajectory with ground truth.

    :param trajectory: Estimated positions (N x 3 array)
    :param est_euler: Estimated Euler angles in degrees (N x 3 array: yaw, pitch, roll)
    :param gt: Ground truth trajectory as a PosePath3D object
    :param start: Start index to select ground truth segment
    :param end: End index to select ground truth segment
    :return: None
    """
    poses_se3 = []
    gt_cut = PosePath3D(poses_se3=gt.poses_se3[start:end])

    for pos, euler in zip(trajectory, est_euler):
        rot = R.from_euler('zyx', euler, degrees=True).as_matrix()  # 3x3 macierz rotacji
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        poses_se3.append(T)
    traj = PosePath3D(poses_se3=poses_se3)
    for relation in [
        PoseRelation.translation_part,
        PoseRelation.rotation_angle_deg,
        PoseRelation.rotation_part,
        PoseRelation.full_transformation
    ]:
        ape_metric = APE(relation)
        ape_metric.process_data((gt_cut, traj))
        stats = ape_metric.get_result().stats
        print(f"APE ({relation.name}): RMSE = {stats['rmse']:.4f}")

    # Relative Pose Error (RPE)
    for relation in [
        PoseRelation.translation_part,
        PoseRelation.rotation_angle_deg,
        PoseRelation.rotation_part,
        PoseRelation.full_transformation
    ]:
        rpe_metric = RPE(relation, delta=1, delta_unit=Unit.frames)
        rpe_metric.process_data((gt_cut, traj))
        stats = rpe_metric.get_result().stats
        print(f"RPE ({relation.name}): RMSE = {stats['rmse']:.4f}")


def metrics_from_file(groundtruth: str, start: int, end: int, file_name: str) -> None:
    """
    Loads trajectory and Euler angle data from file, computes APE and RPE metrics,
    prints total trajectory length and frame count, and plots trajectory and orientation comparison.
    Work fo KITTI dataset!

    :param scene: Scene name used to locate the ground truth file.
    :param start: Start frame index (inclusive).
    :param end: End frame index (exclusive).
    :param file_name: Name of the file in 'saved_trajectories/' to load estimated positions and Euler angles.
    :return: None
    """
    data = np.loadtxt("saved_trajectories//" + file_name, skiprows=1)
    positions = data[:, 0:3][start:end]
    euler_angles = data[:, 3:6][start:end]
    pose = file_interface.read_kitti_poses_file(groundtruth)
    metrics(positions, euler_angles, pose, start, end)
    poses = pose.positions_xyz[start:end]
    length = 0
    for i in range(1, len(poses)):
        p1 = poses[i-1]  # position xyz from matrix 4x4
        p2 = poses[i]
        length += np.linalg.norm(p2 - p1)
    print(f"Total Lenght: {length}")
    print(f"Frames: {end-start}")
    quaternions = pose.orientations_quat_wxyz[start:end]
    quats_xyzw = quaternions[:, [1, 2, 3, 0]]

    gt_rot = R.from_quat(quats_xyzw)
    gt_euler = gt_rot.as_euler('zyx', degrees=True)  # yaw, pitch, roll
    plot_result(positions, poses, euler_angles, gt_euler)


def metrics_from_file_tum(groundtruth: str, start: int, end: int, file_name: str) -> None:
    """
    Loads trajectory and Euler angle data from file, computes APE and RPE metrics,
    prints total trajectory length and frame count, and plots trajectory and orientation comparison.
    Work fo TUM dataset!

    :param scene: Scene name used to locate the ground truth file.
    :param start: Start frame index (inclusive).
    :param end: End frame index (exclusive).
    :param file_name: Name of the file in 'saved_trajectories/' to load estimated positions and Euler angles.
    :return: None
    """
    poses_se3 = []

    positions, euler_angles, time_array = load_trajectory("saved_trajectories//" + file_name)
    positions = positions[start:end]
    euler_angles = euler_angles[start:end]
    # quaternions = R.from_euler("zyx", euler_angles, degrees=True).as_quat()

    # corect_trajectory(
    #          groundtruth / "groundtruth.txt", quaternions, time_array, "rgbd_dataset_freiburg1_floor", positions)
    
    gt_t, gt_euler, time_array_gt = get_gt_tum(start, end, groundtruth)
    for pos, euler in zip(gt_t, gt_euler):
        rot = R.from_euler('zyx', euler, degrees=True).as_matrix()  # 3x3 macierz rotacji
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        poses_se3.append(T)
    pose = PosePath3D(poses_se3=poses_se3)
    metrics(positions, euler_angles, pose, start, end)
    length = 0
    for i in range(1, len(gt_t)):
        p1 = gt_t[i-1]  # pozycja xyz z macierzy 4x4
        p2 = gt_t[i]
        length += np.linalg.norm(p2 - p1)
    print(f"Total Lenght: {length}")
    print(f"Frames: {end-start}")

    plot_result(positions, gt_t, euler_angles, gt_euler)