# /bin/python3

import matplotlib
matplotlib.use('TkAgg') 
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
from pathlib import Path
import argparse
import torch
from tqdm import tqdm

# import superpoint_pytorch
import brevitas_superpoint

if __name__ == "__main__":

    ### Main configuration ###
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maindir",
        type=Path,
        default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint",
    )

    parser.add_argument("--network", type=str, default="/uczelnia/Repositorium/superpoint-fpga/SuperPoint/weights/superpoint_v1.pth")
    
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the network
    print(f"Loading network from {args.network}")
    model = brevitas_superpoint.SuperPointNet_pretrained().eval()
    model.load_state_dict(
        torch.load(args.network, weights_only=True)
    )
    
    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")
    

# FP32 - 4.96 MB
# 8-bit - 1.24 MB
# 4-bit - 0.62 MB
# 3-bit - 0.47 MB
# 4-2-4-bit - 0.33 MB
