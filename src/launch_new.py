import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import processing
from evo.tools import file_interface
from pathlib import Path
import os
from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import time
from trajectory_tools import corect_trajectory, get_gt_tum, plot_result, save_trajectory, save_trajectory_with_euler, metrics, metrics_from_file, metrics_from_file_tum
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def compute_sequence(
    image_folder,
    t_gt,
    start_r,
    matching_type,
    database="tum",
    args=None,
    depth_image_folder=None,
):
    pre_times = []
    net_times = []
    post_times = []
    matching_times = []
    all_times = []
    matches_list = []
    pos_times = []
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".ppm"))]
    )

    if depth_image_folder is not None and args.vo_type == "rgbd":
        depth_image_files = sorted(
            [
                f
                for f in os.listdir(depth_image_folder)
                if f.endswith((".png", ".jpg", ".ppm"))
            ]
        )

    if database == "tum":
        camera = processing.PinholeCamera(
            517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633
        )
    if database == "kitti":
        camera = processing.PinholeCamera(
            fx=707.0912,
            fy=707.0912,
            cx=601.8873,
            cy=183.1104,
            d0=0.0,
            d1=0.0,
            d2=0.0,
            d3=0.0,
            d4=0.0,  # brak danych o dystorsji w P
        )
    # camera = processing.PinholeCamera(520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172)
    K_l = np.array(
        [camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0]
    ).reshape(3, 3)

    start_t = t_gt[0]
    t_start = np.array([[start_t[0]], [start_t[1]], [start_t[2]]])

    # Rotacja z kwaternionu
    r = R.from_euler("zyx", start_r, degrees=True)
    start_R = r.as_matrix()

    first_image = cv2.imread(os.path.join(image_folder, image_files[args.start]))

    first_depth_image = None
    if depth_image_folder is not None and args.vo_type == "rgbd":
        first_depth_image = cv2.imread(
            os.path.join(depth_image_folder, depth_image_files[args.start]),
            cv2.IMREAD_UNCHANGED,
        )
        cv2.imshow("Depth", first_depth_image)

        # first_image = processing.depth_to_rgb(depth_image)
        # first_image = cv2.resize(first_image, (first_image.shape[1], first_image.shape[0]))
    image_size = (first_image.shape[1], first_image.shape[0])
    # image_size = (416, 128)
    # first_image, offset = processing.crop_center(first_image, image_size)
    # camera.cx=camera.cx - offset[0]
    # camera.cy=camera.cy - offset[1]
    scale_x = image_size[0] / first_image.shape[1]
    scale_y = image_size[1] / first_image.shape[0]

    camera.fx = camera.fx * scale_x
    camera.fy = camera.fy * scale_y
    camera.cx = camera.cx * scale_x
    camera.cy = camera.cy * scale_y

    # if database == "tum":
    #     first_image = cv2.resize(first_image, (image_size[0] + 32, image_size[1] + 16))
    #     first_image = cv2.undistort(first_image, K_l, camera.d)
    #     first_image = first_image[8:-8, 16:-16, :]
    # else:
    #     first_image = cv2.resize(first_image, image_size)

    odometry = processing.VisualOdometry(
        args,
        image_size,
        start_R,
        t_start,
        camera,
        matching_type,
        database,
        superpoint_weights=args.maindir / args.network,
        superglue_weights=args.maindir / args.superglue_weights,
    )

    odometry.compute_first_image(first_image, first_depth_image)

    depth_image = None
    for i in tqdm(range(args.start + 1, args.end)):
        # print(f"Processing: {(i-(args.start+1))/(args.end-args.start+1) * 100}%")
        start = time.perf_counter()

        image_file = image_files[i]
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if args.vo_type == "rgbd":
            depth_image = cv2.imread(
                os.path.join(depth_image_folder, depth_image_files[i]),
                cv2.IMREAD_UNCHANGED,
            )

            cv2.imshow("Depth", depth_image)

        skip_frame, times = odometry.compute_pipeline(
            image, t_gt[i - args.start], t_gt[i - args.start - 1], depth_image
        )
        if skip_frame == 0:
            odometry.past_predictions = odometry.present_predictions.copy()

            if args.vo_type == "rgbd":
                odometry.past_depth = odometry.present_depth.copy()
        end = time.perf_counter()

        cv2.imshow("Film", odometry.feature_detection.input_image)

        # Czekaj 30 ms na kolejny obraz (około 30 FPS)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        pre_times.append(times[0])
        net_times.append(times[1])
        post_times.append(times[2])
        matching_times.append(times[3])
        pos_times.append(times[4])

        matches_list.append(times[5])
        all_times.append(end - start)

    num_iterations = len(pre_times) - 1
    avg_pre = sum(pre_times[1:]) / num_iterations * 1000
    avg_net = sum(net_times[1:]) / num_iterations * 1000
    avg_post = sum(post_times[1:]) / num_iterations * 1000
    avg_matching = sum(matching_times[1:]) / num_iterations * 1000
    avg_pos = sum(pos_times[1:]) / num_iterations * 1000

    avg_all = sum(all_times[1:]) / num_iterations * 1000
    avg_matches = sum(matches_list[1:]) / num_iterations

    print(
        f"Średnie czasy (ms): pre: {avg_pre:.6f} | net: {avg_net:.6f} | post: {avg_post:.6f}| mach: {avg_matching:.6f} | pose: {avg_pos:.6f} | all: {avg_all:.6f} | matches: {avg_matches:.6f}"
    )
    return np.array(odometry.trajectory), np.array(odometry.R_list)


if __name__ == "__main__":

    ### Main configuration ###
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maindir",
        type=Path,
        # default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint",
        default=''
    )
    parser.add_argument(
        "--superglue_weights", type=str, default="weights/superglue_indoor.pth"
    )

    parser.add_argument("--vo_type", type=str, default="rgb")  # [rgb, rgbd]
    parser.add_argument("--database", type=str, default="tum")  # [tum, kitti]
    parser.add_argument("--network", type=str, default="weights/superpoint_v1.pth")
    parser.add_argument(
        "--kittidir",
        type=Path,
        # default="/uczelnia/Repositorium/superpoint-fpga/data_odometry_color/dataset",
        default="data/dataset"
    )
    parser.add_argument(
        "--tumdir",
        type=Path,
        # default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint/data",
        default="data"
    )
    
    parser.add_argument("--tum_seq", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--kitti_seq", type=str, default="00")
    parser.add_argument("--kitti_gt", type=Path, default="data/datasets/poses/00.txt")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--viz", action="store_true", default=True)
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    comment = "416x128_SuperPoint_SuperGlue"

    if args.database == "tum":
        file = args.tumdir / args.tum_seq
        gt_t, gt_euler, time_array = get_gt_tum(args.start, args.end, file)
        file_image = file / "rgb"
    if args.database == "kitti":
        groundtruth = args.kittidir / "poses" / f"{args.kitti_seq}.txt"

        pose = file_interface.read_kitti_poses_file(groundtruth)
        gt_t = pose.positions_xyz[args.start : args.end]
        quaternions = pose.orientations_quat_wxyz[args.start : args.end]
        quats_xyzw = quaternions[:, [1, 2, 3, 0]]

        gt_rot = R.from_quat(quats_xyzw)  # scipy używa kolejności: x, y, z, w
        gt_euler = gt_rot.as_euler("zyx", degrees=True)  # yaw, pitch, roll
        file_image = args.kittidir / "sequences" / args.kitti_seq / "image_0"

    trajectory, est_euler = compute_sequence(
        file_image,
        gt_t,
        gt_euler[0],
        "bf",
        args.database,
        args=args,
        depth_image_folder=file / "depth" if args.database == "tum" else None,
    )

    if args.database == "tum":
        quaternions = R.from_euler("zyx", est_euler, degrees=True).as_quat()
        groundtruth = file / "groundtruth.txt"
        corect_trajectory(
            groundtruth, quaternions, time_array, args.tum_seq, trajectory
        )
        save_trajectory(
            trajectory,
            est_euler,
            time_array,
            args.maindir / "saved_trajectories/generated_trajectory.txt",
        )
    if args.database == "kitti":
        metrics(trajectory, est_euler, pose, args, args.start, args.end)
    save_trajectory_with_euler(
        args.maindir
        / f"saved_trajectories/{args.database}_{args.kitti_seq}_{args.start}_{args.end}_{comment}.txt",
        trajectory,
        est_euler,
    )
    if args.viz:
        plot_result(trajectory, gt_t, est_euler, gt_euler)

if args.database == "kitti":
    metrics_from_file(args.kittidir / "poses" / f"{args.kitti_seq}.txt", comment, args)

# scene = "00"
# database = 'tum'
# start = 0
# end = 200
# comment = "test"
# file_name = "tum_not_int8t.txt"
# data1 = np.loadtxt("saved_trajectories//00_soft (1).txt", skiprows=1)
# SuperPoint_bf = data1[:, 0:3][start:end]
# metrics_from_file_tum(file_name)