# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class SuperPoint_short(torch.nn.Module):
    def __init__(self):
        super(SuperPoint_short, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SuperPoint_short::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/VGGBlock[0]/Conv2d[conv]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/VGGBlock[0]/ReLU[activation]/input.5
        self.module_3 = py_nndct.nn.BatchNorm(num_features=64, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/VGGBlock[0]/BatchNorm2d[bn]/input.7
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/VGGBlock[1]/Conv2d[conv]/input.9
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/VGGBlock[1]/ReLU[activation]/input.11
        self.module_6 = py_nndct.nn.BatchNorm(num_features=64, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/VGGBlock[1]/BatchNorm2d[bn]/1947
        self.module_7 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[0]/MaxPool2d[2]/input.13
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/VGGBlock[0]/Conv2d[conv]/input.15
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/VGGBlock[0]/ReLU[activation]/input.17
        self.module_10 = py_nndct.nn.BatchNorm(num_features=64, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/VGGBlock[0]/BatchNorm2d[bn]/input.19
        self.module_11 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/VGGBlock[1]/Conv2d[conv]/input.21
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/VGGBlock[1]/ReLU[activation]/input.23
        self.module_13 = py_nndct.nn.BatchNorm(num_features=64, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/VGGBlock[1]/BatchNorm2d[bn]/2011
        self.module_14 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[1]/MaxPool2d[2]/input.25
        self.module_15 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/VGGBlock[0]/Conv2d[conv]/input.27
        self.module_16 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/VGGBlock[0]/ReLU[activation]/input.29
        self.module_17 = py_nndct.nn.BatchNorm(num_features=128, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/VGGBlock[0]/BatchNorm2d[bn]/input.31
        self.module_18 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/VGGBlock[1]/Conv2d[conv]/input.33
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/VGGBlock[1]/ReLU[activation]/input.35
        self.module_20 = py_nndct.nn.BatchNorm(num_features=128, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/VGGBlock[1]/BatchNorm2d[bn]/2075
        self.module_21 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[2]/MaxPool2d[2]/input.37
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[3]/VGGBlock[0]/Conv2d[conv]/input.39
        self.module_23 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[3]/VGGBlock[0]/ReLU[activation]/input.41
        self.module_24 = py_nndct.nn.BatchNorm(num_features=128, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[3]/VGGBlock[0]/BatchNorm2d[bn]/input.43
        self.module_25 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[3]/VGGBlock[1]/Conv2d[conv]/input.45
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[3]/VGGBlock[1]/ReLU[activation]/input.47
        self.module_27 = py_nndct.nn.BatchNorm(num_features=128, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[backbone]/Sequential[3]/VGGBlock[1]/BatchNorm2d[bn]/input.49
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[descriptor]/VGGBlock[0]/Conv2d[conv]/input.51
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[descriptor]/VGGBlock[0]/ReLU[activation]/input.53
        self.module_30 = py_nndct.nn.BatchNorm(num_features=256, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[descriptor]/VGGBlock[0]/BatchNorm2d[bn]/input.55
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[descriptor]/VGGBlock[1]/Conv2d[conv]/input.57
        self.module_32 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[detector]/VGGBlock[0]/Conv2d[conv]/input.59
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #SuperPoint_short::SuperPoint_short/Sequential[detector]/VGGBlock[0]/ReLU[activation]/input.61
        self.module_34 = py_nndct.nn.BatchNorm(num_features=256, eps=0.0, momentum=0.1) #SuperPoint_short::SuperPoint_short/Sequential[detector]/VGGBlock[0]/BatchNorm2d[bn]/input.63
        self.module_35 = py_nndct.nn.Conv2d(in_channels=256, out_channels=65, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SuperPoint_short::SuperPoint_short/Sequential[detector]/VGGBlock[1]/Conv2d[conv]/input

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        output_module_0 = self.module_14(output_module_0)
        output_module_0 = self.module_15(output_module_0)
        output_module_0 = self.module_16(output_module_0)
        output_module_0 = self.module_17(output_module_0)
        output_module_0 = self.module_18(output_module_0)
        output_module_0 = self.module_19(output_module_0)
        output_module_0 = self.module_20(output_module_0)
        output_module_0 = self.module_21(output_module_0)
        output_module_0 = self.module_22(output_module_0)
        output_module_0 = self.module_23(output_module_0)
        output_module_0 = self.module_24(output_module_0)
        output_module_0 = self.module_25(output_module_0)
        output_module_0 = self.module_26(output_module_0)
        output_module_0 = self.module_27(output_module_0)
        output_module_28 = self.module_28(output_module_0)
        output_module_28 = self.module_29(output_module_28)
        output_module_28 = self.module_30(output_module_28)
        output_module_28 = self.module_31(output_module_28)
        output_module_32 = self.module_32(output_module_0)
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(output_module_32)
        return output_module_32,output_module_28
