import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas import config

from brevitas.core.quant import QuantType
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFloat,
)
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Int16Bias

config.IGNORE_MISSING_KEYS = True
from types import SimpleNamespace

class UIntActQuant(Uint8ActPerTensorFloat):
    """Floating point scales restricted to 2e-5 (to avoid floating-point shenanigans during streamlining)"""
    scaling_min_val = 2e-5
    restrict_scaling_type = RestrictValueType.LOG_FP


class SuperPointNet_pretrained(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""
    default_conf = {
            "nms_radius": 4,
            "max_num_keypoints": 500,
            "detection_threshold": 0.005,
            "remove_borders": 4,
            "descriptor_dim": 256,
            "channels": [64, 64, 128, 128, 256],
        }
    def __init__(self, **conf):
        
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        
        super(SuperPointNet_pretrained, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)

        # qnn.QuantReLU(quant_type=QuantType.F),
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        
        bit_width = 8
        first_last_bit_width = 8
        # Shared Encoder.
        self.conv1a = QuantConv2d(
            1,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=first_last_bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu1a = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv1b = QuantConv2d(
            c1,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu1b = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv2a = QuantConv2d(
            c1,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu2a = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv2b = QuantConv2d(
            c2,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu2b = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv3a = QuantConv2d(
            c2,
            c3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu3a = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv3b = QuantConv2d(
            c3,
            c3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu3b = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv4a = QuantConv2d(
            c3,
            c4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu4a = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        self.conv4b = QuantConv2d(
            c4,
            c4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.relu4b = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=bit_width
        )
        # Detector Head.
        self.convPa = QuantConv2d(
            c4,
            c5,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.reluPa = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=first_last_bit_width
        )
        self.convPb = QuantConv2d(
            c5,
            65,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            weight_bit_width=first_last_bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        # Descriptor Head.
        self.convDa = QuantConv2d(
            c4,
            c5,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )
        self.reluDa = QuantReLU(
            act_quant=UIntActQuant,
            bit_width=first_last_bit_width
        )
        self.convDb = QuantConv2d(
            c5,
            d1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            weight_bit_width=first_last_bit_width,
            # weight_quant=Int8WeightPerTensorFixedPoint,
            # bias_quant=Int16Bias,
        )

    def forward(self, x):
        """Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """

        # x = self.quant_inp(x)
        # Shared Encoder.
        x = self.relu1a(self.conv1a(x))
        x = self.relu1b(self.conv1b(x))
        x = self.pool(x)
        x = self.relu2a(self.conv2a(x))
        x = self.relu2b(self.conv2b(x))
        x = self.pool(x)
        x = self.relu3a(self.conv3a(x))
        x = self.relu3b(self.conv3b(x))
        x = self.pool(x)
        x = self.relu4a(self.conv4a(x))
        x = self.relu4b(self.conv4b(x))
        # Detector Head.
        cPa = self.reluPa(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.reluDa(self.convDa(x))
        desc = self.convDb(cDa)
        
        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        
        
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        output = {"semi": semi, "desc": desc}
        return scores, desc


class SuperPointNet(nn.Module):
    """ Pytorch definition of SuperPoint Network. """
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
        self.conf = conf
        self.stride = 2 ** (len(self.conf["channels"]) - 2)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.relu = torch.nn.ReLU(inplace=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.relu8 = torch.nn.ReLU(inplace=True)
        self.relu9 = torch.nn.ReLU(inplace=True)
        self.relu10 = torch.nn.ReLU(inplace=True)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(
            1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(
            c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(
            c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(
            c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(
            c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(
            c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(
            c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(
            c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(
            c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(
            c5, d1, kernel_size=1, stride=1, padding=0)

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
        x = self.quant(x)
        x = self.relu1(self.conv1a(x))
        x = self.relu2(self.conv1b(x))
        x = self.pool1(x)
        x = self.relu3(self.conv2a(x))
        x = self.relu4(self.conv2b(x))
        x = self.pool2(x)
        x = self.relu5(self.conv3a(x))
        x = self.relu6(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu7(self.conv4a(x))
        x = self.relu8(self.conv4b(x))
        # Detector Head.
        cPa = self.relu9(self.convPa(x))
        semi = self.convPb(cPa)
        scores = torch.nn.functional.softmax(semi, 1)
        # Descriptor Head.
        cDa = self.relu10(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        desc = torch.nn.functional.normalize(desc, p=2, dim=1, eps=1e-6)
        return scores, desc
