# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class SuperPoint(torch.nn.Module):
    def __init__(self):
        super(SuperPoint, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SuperPoint::input_0
        self.module_1 = py_nndct.nn.Module('const') #SuperPoint::SuperPoint/330
        self.module_2 = py_nndct.nn.Module('const') #SuperPoint::SuperPoint/333
        self.module_3 = py_nndct.nn.Module('const') #SuperPoint::410
        self.module_4 = py_nndct.nn.Module('const') #SuperPoint::428
        self.module_5 = py_nndct.nn.Module('const') #SuperPoint::441
        self.module_6 = py_nndct.nn.Module('const') #SuperPoint::459
        self.module_7 = py_nndct.nn.Module('const') #SuperPoint::SuperPoint/524
        self.module_8 = py_nndct.nn.Module('const') #SuperPoint::527
        self.module_9 = py_nndct.nn.Module('const') #SuperPoint::SuperPoint/529
        self.module_10 = py_nndct.nn.Module('const') #SuperPoint::SuperPoint/531
        self.module_11 = py_nndct.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/VGGBlock[0]/Conv2d[conv]/input.2
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/VGGBlock[0]/ReLU[activation]/input.3
        self.module_13 = py_nndct.nn.Module('batch_norm',num_features=64, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/VGGBlock[0]/BatchNorm2d[bn]/input.4
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/VGGBlock[1]/Conv2d[conv]/input.5
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/VGGBlock[1]/ReLU[activation]/input.6
        self.module_16 = py_nndct.nn.Module('batch_norm',num_features=64, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/VGGBlock[1]/BatchNorm2d[bn]/116
        self.module_17 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[0]/MaxPool2d[2]/input.7
        self.module_18 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/VGGBlock[0]/Conv2d[conv]/input.8
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/VGGBlock[0]/ReLU[activation]/input.9
        self.module_20 = py_nndct.nn.Module('batch_norm',num_features=64, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/VGGBlock[0]/BatchNorm2d[bn]/input.10
        self.module_21 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/VGGBlock[1]/Conv2d[conv]/input.11
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/VGGBlock[1]/ReLU[activation]/input.12
        self.module_23 = py_nndct.nn.Module('batch_norm',num_features=64, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/VGGBlock[1]/BatchNorm2d[bn]/154
        self.module_24 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[1]/MaxPool2d[2]/input.13
        self.module_25 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/VGGBlock[0]/Conv2d[conv]/input.14
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/VGGBlock[0]/ReLU[activation]/input.15
        self.module_27 = py_nndct.nn.Module('batch_norm',num_features=128, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/VGGBlock[0]/BatchNorm2d[bn]/input.16
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/VGGBlock[1]/Conv2d[conv]/input.17
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/VGGBlock[1]/ReLU[activation]/input.18
        self.module_30 = py_nndct.nn.Module('batch_norm',num_features=128, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/VGGBlock[1]/BatchNorm2d[bn]/192
        self.module_31 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[2]/MaxPool2d[2]/input.19
        self.module_32 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[3]/VGGBlock[0]/Conv2d[conv]/input.20
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[3]/VGGBlock[0]/ReLU[activation]/input.21
        self.module_34 = py_nndct.nn.Module('batch_norm',num_features=128, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[3]/VGGBlock[0]/BatchNorm2d[bn]/input.22
        self.module_35 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[3]/VGGBlock[1]/Conv2d[conv]/input.23
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[3]/VGGBlock[1]/ReLU[activation]/input.24
        self.module_37 = py_nndct.nn.Module('batch_norm',num_features=128, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[backbone]/Sequential[3]/VGGBlock[1]/BatchNorm2d[bn]/input.25
        self.module_38 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[descriptor]/VGGBlock[0]/Conv2d[conv]/input.26
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[descriptor]/VGGBlock[0]/ReLU[activation]/input.27
        self.module_40 = py_nndct.nn.Module('batch_norm',num_features=256, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[descriptor]/VGGBlock[0]/BatchNorm2d[bn]/input.28
        self.module_41 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[descriptor]/VGGBlock[1]/Conv2d[conv]/input.29
        self.module_43 = py_nndct.nn.Module('normalize') #SuperPoint::SuperPoint/265
        self.module_44 = py_nndct.nn.Module('clamp') #SuperPoint::SuperPoint/267
        self.module_45 = py_nndct.nn.Module('expand_as') #SuperPoint::SuperPoint/268
        self.module_46 = py_nndct.nn.Module('elemwise_div') #SuperPoint::SuperPoint/269
        self.module_47 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[detector]/VGGBlock[0]/Conv2d[conv]/input.30
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #SuperPoint::SuperPoint/Sequential[detector]/VGGBlock[0]/ReLU[activation]/input.31
        self.module_49 = py_nndct.nn.Module('batch_norm',num_features=256, eps=0.0, momentum=0.1) #SuperPoint::SuperPoint/Sequential[detector]/VGGBlock[0]/BatchNorm2d[bn]/input.32
        self.module_50 = py_nndct.nn.Conv2d(in_channels=256, out_channels=65, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SuperPoint::SuperPoint/Sequential[detector]/VGGBlock[1]/Conv2d[conv]/input
        self.module_52 = py_nndct.nn.Module('softmax',dim=1) #SuperPoint::SuperPoint/303
        self.module_53 = py_nndct.nn.strided_slice() #SuperPoint::SuperPoint/308
        self.module_54 = py_nndct.nn.Module('shape') #SuperPoint::SuperPoint/315
        self.module_55 = py_nndct.nn.Module('shape') #SuperPoint::SuperPoint/317
        self.module_56 = py_nndct.nn.Module('tensor') #SuperPoint::SuperPoint/318
        self.module_57 = py_nndct.nn.Module('shape') #SuperPoint::SuperPoint/320
        self.module_58 = py_nndct.nn.Module('tensor') #SuperPoint::SuperPoint/321
        self.module_59 = py_nndct.nn.Module('permute') #SuperPoint::SuperPoint/323
        self.module_60 = py_nndct.nn.Module('reshape') #SuperPoint::SuperPoint/327
        self.module_61 = py_nndct.nn.Module('permute') #SuperPoint::SuperPoint/329
        self.module_62 = py_nndct.nn.Module('mul') #SuperPoint::SuperPoint/331
        self.module_63 = py_nndct.nn.Int() #SuperPoint::SuperPoint/332
        self.module_64 = py_nndct.nn.Module('mul') #SuperPoint::SuperPoint/334
        self.module_65 = py_nndct.nn.Int() #SuperPoint::SuperPoint/335
        self.module_66 = py_nndct.nn.Module('reshape') #SuperPoint::SuperPoint/scores.2
        self.module_67 = py_nndct.nn.Module('aten::zeros_like') #SuperPoint::SuperPoint/343
        self.module_68 = py_nndct.nn.MaxPool2d(kernel_size=[11, 11], stride=[1, 1], padding=[5, 5], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/349
        self.module_69 = py_nndct.nn.Module('equal') #SuperPoint::SuperPoint/350
        self.module_70 = py_nndct.nn.Module('cast') #SuperPoint::SuperPoint/355
        self.module_71 = py_nndct.nn.MaxPool2d(kernel_size=[11, 11], stride=[1, 1], padding=[5, 5], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/361
        self.module_72 = py_nndct.nn.Module('aten::gt') #SuperPoint::SuperPoint/363
        self.module_73 = py_nndct.nn.Module('aten::where') #SuperPoint::SuperPoint/supp_scores.1
        self.module_74 = py_nndct.nn.MaxPool2d(kernel_size=[11, 11], stride=[1, 1], padding=[5, 5], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/370
        self.module_75 = py_nndct.nn.Module('equal') #SuperPoint::SuperPoint/371
        self.module_76 = py_nndct.nn.Module('aten::bitwise_not') #SuperPoint::SuperPoint/372
        self.module_77 = py_nndct.nn.Module('aten::__and__') #SuperPoint::SuperPoint/373
        self.module_78 = py_nndct.nn.Module('aten::__or__') #SuperPoint::SuperPoint/374
        self.module_79 = py_nndct.nn.Module('cast') #SuperPoint::SuperPoint/379
        self.module_80 = py_nndct.nn.MaxPool2d(kernel_size=[11, 11], stride=[1, 1], padding=[5, 5], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/385
        self.module_81 = py_nndct.nn.Module('aten::gt') #SuperPoint::SuperPoint/387
        self.module_82 = py_nndct.nn.Module('aten::where') #SuperPoint::SuperPoint/supp_scores
        self.module_83 = py_nndct.nn.MaxPool2d(kernel_size=[11, 11], stride=[1, 1], padding=[5, 5], dilation=[1, 1], ceil_mode=False) #SuperPoint::SuperPoint/394
        self.module_84 = py_nndct.nn.Module('equal') #SuperPoint::SuperPoint/395
        self.module_85 = py_nndct.nn.Module('aten::bitwise_not') #SuperPoint::SuperPoint/396
        self.module_86 = py_nndct.nn.Module('aten::__and__') #SuperPoint::SuperPoint/397
        self.module_87 = py_nndct.nn.Module('aten::__or__') #SuperPoint::SuperPoint/398
        self.module_88 = py_nndct.nn.Module('aten::where') #SuperPoint::SuperPoint/399
        self.module_93 = py_nndct.nn.Module('squeeze') #SuperPoint::SuperPoint/scores
        self.module_94 = py_nndct.nn.Module('aten::gt') #SuperPoint::SuperPoint/465
        self.module_95 = py_nndct.nn.Module('aten::where') #SuperPoint::SuperPoint/467
        self.module_96 = py_nndct.nn.Module('stack') #SuperPoint::SuperPoint/471
        self.module_97 = py_nndct.nn.Module('aten::flip') #SuperPoint::SuperPoint/473
        self.module_98 = py_nndct.nn.Module('cast') #SuperPoint::SuperPoint/keypoints
        self.module_99 = py_nndct.nn.Module('cast') #SuperPoint::SuperPoint/486
        self.module_100 = py_nndct.nn.Module('cast') #SuperPoint::SuperPoint/494
        self.module_101 = py_nndct.nn.Index() #SuperPoint::SuperPoint/s
        self.module_102 = py_nndct.nn.Module('aten::topk') #SuperPoint::SuperPoint/501
        self.module_103 = py_nndct.nn.Module('cast') #SuperPoint::SuperPoint/510
        self.module_104 = py_nndct.nn.Index() #SuperPoint::SuperPoint/512
        self.module_105 = py_nndct.nn.Module('unsqueeze') #SuperPoint::SuperPoint/514
        self.module_106 = py_nndct.nn.Module('select') #SuperPoint::SuperPoint/517
        self.module_107 = py_nndct.nn.Module('unsqueeze') #SuperPoint::SuperPoint/descriptors
        self.module_108 = py_nndct.nn.Module('shape') #SuperPoint::SuperPoint/521
        self.module_109 = py_nndct.nn.Module('shape') #SuperPoint::SuperPoint/523
        self.module_110 = py_nndct.nn.Add() #SuperPoint::SuperPoint/526
        self.module_111 = py_nndct.nn.Module('elemwise_div') #SuperPoint::SuperPoint/528
        self.module_112 = py_nndct.nn.Module('elemwise_mul') #SuperPoint::SuperPoint/530
        self.module_113 = py_nndct.nn.Sub() #SuperPoint::SuperPoint/533
        self.module_114 = py_nndct.nn.Module('reshape') #SuperPoint::SuperPoint/538
        self.module_115 = py_nndct.nn.Module('grid_sample') #SuperPoint::SuperPoint/542
        self.module_116 = py_nndct.nn.Module('reshape') #SuperPoint::SuperPoint/545
        self.module_117 = py_nndct.nn.Module('normalize') #SuperPoint::SuperPoint/549
        self.module_118 = py_nndct.nn.Module('clamp') #SuperPoint::SuperPoint/551
        self.module_119 = py_nndct.nn.Module('expand_as') #SuperPoint::SuperPoint/552
        self.module_120 = py_nndct.nn.Module('elemwise_div') #SuperPoint::SuperPoint/553
        self.module_121 = py_nndct.nn.Module('squeeze') #SuperPoint::SuperPoint/555
        self.module_122 = py_nndct.nn.Module('transpose') #SuperPoint::SuperPoint/558

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(dtype=torch.long, device='cpu', data=8)
        self.output_module_2 = self.module_2(dtype=torch.long, device='cpu', data=8)
        self.output_module_3 = self.module_3(dtype=torch.float, device='cpu', data=[[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]])
        self.output_module_4 = self.module_4(dtype=torch.float, device='cpu', data=[[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0]])
        self.output_module_5 = self.module_5(dtype=torch.float, device='cpu', data=[[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]])
        self.output_module_6 = self.module_6(dtype=torch.float, device='cpu', data=[[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0,-1.0]])
        self.output_module_7 = self.module_7(dtype=torch.double, device='cpu', data=0.5)
        self.output_module_8 = self.module_8(dtype=torch.float, device='cpu', data=[296.0,200.0])
        self.output_module_9 = self.module_9(dtype=torch.long, device='cpu', data=2)
        self.output_module_10 = self.module_10(dtype=torch.long, device='cpu', data=1)
        self.output_module_11 = self.module_11(self.output_module_0)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_14 = self.module_14(self.output_module_13)
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_17 = self.module_17(self.output_module_16)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_19 = self.module_19(self.output_module_18)
        self.output_module_20 = self.module_20(self.output_module_19)
        self.output_module_21 = self.module_21(self.output_module_20)
        self.output_module_22 = self.module_22(self.output_module_21)
        self.output_module_23 = self.module_23(self.output_module_22)
        self.output_module_24 = self.module_24(self.output_module_23)
        self.output_module_25 = self.module_25(self.output_module_24)
        self.output_module_26 = self.module_26(self.output_module_25)
        self.output_module_27 = self.module_27(self.output_module_26)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_29 = self.module_29(self.output_module_28)
        self.output_module_30 = self.module_30(self.output_module_29)
        self.output_module_31 = self.module_31(self.output_module_30)
        self.output_module_32 = self.module_32(self.output_module_31)
        self.output_module_33 = self.module_33(self.output_module_32)
        self.output_module_34 = self.module_34(self.output_module_33)
        self.output_module_35 = self.module_35(self.output_module_34)
        self.output_module_36 = self.module_36(self.output_module_35)
        self.output_module_37 = self.module_37(self.output_module_36)
        self.output_module_38 = self.module_38(self.output_module_37)
        self.output_module_39 = self.module_39(self.output_module_38)
        self.output_module_40 = self.module_40(self.output_module_39)
        self.output_module_41 = self.module_41(self.output_module_40)
        self.output_module_43 = self.module_43(dim=[1], p='fro', input=self.output_module_41, keepdim=True)
        self.output_module_44 = self.module_44(input=self.output_module_43, min=1e-12)
        self.output_module_45 = self.module_45(input=self.output_module_44, other=self.output_module_41)
        self.output_module_46 = self.module_46(input=self.output_module_41, other=self.output_module_45)
        self.output_module_47 = self.module_47(self.output_module_37)
        self.output_module_48 = self.module_48(self.output_module_47)
        self.output_module_49 = self.module_49(self.output_module_48)
        self.output_module_50 = self.module_50(self.output_module_49)
        self.output_module_52 = self.module_52(self.output_module_50)
        self.output_module_53 = self.module_53(end=[2147483647,-1,2147483647,2147483647], step=[1,1,1,1], input=self.output_module_52, start=[0,0,0,0])
        self.output_module_54 = self.module_54(input=self.output_module_53, dim=0)
        self.output_module_55 = self.module_55(input=self.output_module_53, dim=2)
        self.output_module_56 = self.module_56(dtype=torch.int, device='cpu', data=self.output_module_55)
        self.output_module_57 = self.module_57(input=self.output_module_53, dim=3)
        self.output_module_58 = self.module_58(dtype=torch.int, device='cpu', data=self.output_module_57)
        self.output_module_59 = self.module_59(input=self.output_module_53, dims=[0,2,3,1])
        self.output_module_60 = self.module_60(input=self.output_module_59, size=[self.output_module_54,self.output_module_55,self.output_module_57,8,8])
        self.output_module_61 = self.module_61(input=self.output_module_60, dims=[0,1,3,2,4])
        self.output_module_62 = self.module_62(input=self.output_module_56, other=self.output_module_1)
        self.output_module_63 = self.module_63(input=self.output_module_62)
        self.output_module_64 = self.module_64(input=self.output_module_58, other=self.output_module_2)
        self.output_module_65 = self.module_65(input=self.output_module_64)
        self.output_module_66 = self.module_66(input=self.output_module_61, size=[self.output_module_54,self.output_module_63,self.output_module_65])
        self.output_module_67 = self.module_67(pin_memory=False, dtype=torch.float, device='cpu', input=self.output_module_66)
        self.output_module_68 = self.module_68(self.output_module_66)
        self.output_module_69 = self.module_69(input=self.output_module_66, other=self.output_module_68)
        self.output_module_70 = self.module_70(input=self.output_module_69, dtype=torch.float)
        self.output_module_71 = self.module_71(self.output_module_70)
        self.output_module_72 = self.module_72(input=self.output_module_71, other=0)
        self.output_module_73 = self.module_73(condition=self.output_module_72, input=self.output_module_67, other=self.output_module_66)
        self.output_module_74 = self.module_74(self.output_module_73)
        self.output_module_75 = self.module_75(input=self.output_module_73, other=self.output_module_74)
        self.output_module_76 = self.module_76(input=self.output_module_72, )
        self.output_module_77 = self.module_77(input=self.output_module_75, other=self.output_module_76)
        self.output_module_78 = self.module_78(input=self.output_module_69, other=self.output_module_77)
        self.output_module_79 = self.module_79(input=self.output_module_78, dtype=torch.float)
        self.output_module_80 = self.module_80(self.output_module_79)
        self.output_module_81 = self.module_81(input=self.output_module_80, other=0)
        self.output_module_82 = self.module_82(condition=self.output_module_81, input=self.output_module_67, other=self.output_module_66)
        self.output_module_83 = self.module_83(self.output_module_82)
        self.output_module_84 = self.module_84(input=self.output_module_82, other=self.output_module_83)
        self.output_module_85 = self.module_85(input=self.output_module_81, )
        self.output_module_86 = self.module_86(input=self.output_module_84, other=self.output_module_85)
        self.output_module_87 = self.module_87(input=self.output_module_78, other=self.output_module_86)
        self.output_module_88 = self.module_88(condition=self.output_module_87, input=self.output_module_66, other=self.output_module_67)
        self.output_module_88[0:9223372036854775807:1,0:4:1,0:2147483647:1] = self.output_module_3
        self.output_module_88[0:9223372036854775807:1,0:9223372036854775807:1,0:4:1] = self.output_module_4
        self.output_module_88[0:9223372036854775807:1,-4:9223372036854775807:1,0:2147483647:1] = self.output_module_5
        self.output_module_88[0:9223372036854775807:1,0:9223372036854775807:1,-4:9223372036854775807:1] = self.output_module_6
        self.output_module_93 = self.module_93(dim=(0), input=self.output_module_88)
        self.output_module_94 = self.module_94(input=self.output_module_93, other=0.005)
        self.output_module_95_0,self.output_module_95_1 = self.module_95(condition=self.output_module_94)
        self.output_module_96 = self.module_96(tensors=[self.output_module_95_0,self.output_module_95_1], dim=-1)
        self.output_module_97 = self.module_97(input=self.output_module_96, dims=[1])
        self.output_module_98 = self.module_98(input=self.output_module_97, dtype=torch.float)
        self.output_module_99 = self.module_99(input=self.output_module_95_0, dtype=torch.int64)
        self.output_module_100 = self.module_100(input=self.output_module_95_1, dtype=torch.int64)
        self.output_module_101 = self.module_101(index=[self.output_module_99,self.output_module_100], input=self.output_module_93)
        self.output_module_102_0,self.output_module_102_1 = self.module_102(input=self.output_module_101, sorted=True, dim=0, k=500, largest=True)
        self.output_module_103 = self.module_103(input=self.output_module_102_1, dtype=torch.int64)
        self.output_module_104 = self.module_104(index=[self.output_module_103], input=self.output_module_98)
        self.output_module_105 = self.module_105(dim=0, input=self.output_module_104)
        self.output_module_106 = self.module_106(index=0, dim=0, input=self.output_module_46)
        self.output_module_107 = self.module_107(dim=0, input=self.output_module_106)
        self.output_module_108 = self.module_108(input=self.output_module_107, dim=0)
        self.output_module_109 = self.module_109(input=self.output_module_107, dim=1)
        self.output_module_110 = self.module_110(alpha=1, input=self.output_module_105, other=self.output_module_7)
        self.output_module_111 = self.module_111(input=self.output_module_110, other=self.output_module_8)
        self.output_module_112 = self.module_112(input=self.output_module_111, other=self.output_module_9)
        self.output_module_113 = self.module_113(alpha=1, input=self.output_module_112, other=self.output_module_10)
        self.output_module_114 = self.module_114(input=self.output_module_113, size=[self.output_module_108,1,-1,2])
        self.output_module_115 = self.module_115(input=self.output_module_107, grid=self.output_module_114, mode='bilinear', padding_mode='zeros', align_corners=False)
        self.output_module_116 = self.module_116(input=self.output_module_115, size=[self.output_module_108,self.output_module_109,-1])
        self.output_module_117 = self.module_117(dim=[1], p='fro', input=self.output_module_116, keepdim=True)
        self.output_module_118 = self.module_118(input=self.output_module_117, min=1e-12)
        self.output_module_119 = self.module_119(input=self.output_module_118, other=self.output_module_116)
        self.output_module_120 = self.module_120(input=self.output_module_116, other=self.output_module_119)
        self.output_module_121 = self.module_121(dim=(0), input=self.output_module_120)
        self.output_module_122 = self.module_122(dim1=1, dim0=0, input=self.output_module_121)
        return self.output_module_104,self.output_module_102_0,self.output_module_122
