import time
import torch.nn as nn
import torch
from collections import OrderedDict
from types import SimpleNamespace

""" 
PyTorch implementation of the SuperPoint model,
derived from the TensorFlow re-implementation (2018).
Authors: Rémi Pautrat, Paul-Edouard Sarlin
Source: https://github.com/rpautrat/SuperPoint
Weights: weights\superpoint_v6_from_tf.pth

"""
def sample_descriptors(keypoints, descriptors, s: int = 8):
    """
    Interpolates descriptors at keypoint locations.
    
    Args:
        keypoints (Tensor): [N, 2] tensor with (x, y) keypoint coordinates.
        descriptors (Tensor): [1, C, Hc, Wc] dense descriptor map.
        s (int): Stride of the output feature map with respect to input image.

    Returns:
        Tensor: [C, N] interpolated descriptors at keypoint locations.
    """
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def batched_nms(scores, nms_radius: int):
    """
    Performs non-maximum suppression (NMS) on a dense score map using max pooling.

    Args:
        scores (Tensor): [B, 1, H, W] score maps.
        dist_thresh (int): Distance threshold for NMS suppression window.

    Returns:
        Tensor: NMS-suppressed score maps of the same shape.
    """
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        bn = nn.BatchNorm2d(c_out, eps=0.001)

        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("bn", bn),
                    ("activation", activation),
                ]
            )
        )


class SuperPoint(nn.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": 500,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

    def forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB to gray
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode the detection scores
        scores = self.detector(features)

        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )

        scores = batched_nms(scores, self.conf.nms_radius)
        lol = scores.numpy()

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

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
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }


class SuperPoint_short(nn.Module):
    """
    Modified version with adjusted post-processing.

    Some elements have been moved outside the network's forward pass 
    into a separate function to enable better comparison with the quantized model, 
    as certain post-processing operations are not supported during quantization.
    """
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": 500,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

    def forward(self, data):
        image = data

        features = self.backbone(image)

        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode the detection scores
        scores = self.detector(features)

        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        return scores, descriptors_dense
    
    def post_processing(scores, descriptors_dense):
        """
        Performs post-processing on the raw network outputs to extract keypoints and their descriptors.

        Steps:
        1. Applies Non-Maximum Suppression (NMS) with a fixed radius to filter out non-local maxima.
        2. Selects keypoints with confidence score above a threshold (0.005).
        3. Converts keypoint coordinates to (x, y) format and casts them to float.
        4. Normalizes the dense descriptor map.
        5. Samples descriptors at the selected keypoint locations.

        Args:
            scores (torch.Tensor): Heatmap with confidence scores for keypoints, shape [1, H, W].
            descriptors_dense (torch.Tensor): Dense descriptor map from the network, shape [1, C, Hc, Wc].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - scores_all (torch.Tensor): Confidence scores of the selected keypoints, shape [N].
                - descriptors (torch.Tensor): Corresponding descriptors, shape [C, N].
        """
        scores = batched_nms(scores, 4)
        scores = scores.squeeze(0)
        idxs = torch.where(scores > 0.005)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        descriptors_dense = torch.nn.functional.normalize(descriptors_dense, p=2, dim=1)

        d = sample_descriptors(keypoints_all[None], descriptors_dense[0, None], 8)
        descriptors = d.squeeze(0).transpose(0, 1)
        return scores_all, descriptors
        
class SuperPoint_short_quant(nn.Module):
    """
    Version of model with addiction quantization functions.
    """
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": 500,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = conf
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf["channels"]) - 2)
        channels = [1, *self.conf["channels"][:-1]]

        # Definicja QuantStub
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf["channels"][-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf["descriptor_dim"], 1, relu=False),
        )

    def forward(self, data):
        # Kwantyzacja wejścia
        image = self.quant(data)

        features = self.backbone(image)
        descriptors_dense = self.descriptor(features)

        # Decode the detection scores
        scores = self.detector(features)
        # Dekwantyzacja przed wyjściem
        return self.dequant(scores), self.dequant(descriptors_dense)
    
class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. 
        Source: https://github.com/magicleap/SuperPointPretrainedNetwork
        Version of model without batchNorm.
        Weights: weights\superpoint_v1.pth
    """
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": 500,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }
    def __init__(self, **conf):
        super(SuperPointNet, self).__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 1 x H x W.
        Output
        semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        start = time.perf_counter()

        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        # dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        # desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc