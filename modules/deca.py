import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle

torch.backends.cudnn.benchmark = True


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)
        x2 = self.avgpool(x1)
        x2 = x2.view(x2.size(0), -1)
        # x = self.fc(x)
        ## x2: [bz, 2048] for shape
        ## x1: [bz, 2048, 7, 7] for texture
        return x2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def copy_parameter_from_resnet(model, resnet_dict):
    cur_state_dict = model.state_dict()
    # import ipdb; ipdb.set_trace()
    for name, param in list(resnet_dict.items())[0:None]:
        if name not in cur_state_dict:
            # print(name, ' not available in reconstructed resnet')
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        try:
            cur_state_dict[name].copy_(param)
        except:
            # print(name, ' is inconsistent!')
            continue
    # print('copy resnet state dict finished!')
    # import ipdb; ipdb.set_trace()


def load_ResNet50Model():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet50(pretrained=False).state_dict())
    return model


def load_ResNet101Model():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet101(pretrained=True).state_dict())
    return model


def load_ResNet152Model():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet152(pretrained=True).state_dict())
    return model


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = load_ResNet50Model()  # out: 2048
        ### regressor
        self.layers = nn.Sequential(nn.Linear(feature_size, 1024), nn.ReLU(), nn.Linear(1024, outsize))
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters


class DECA(nn.Module):
    def __init__(self):
        super(DECA, self).__init__()

        # self.device = "cuda"
        self.pretrained_modelpath = "pretrained/deca_model.tar"
        self.param_dict = {"shape": 100, "tex": 50, "exp": 50, "cam": 3, "pose": 6, "light": 27}
        self.create_model()

    def copy_state_dict(self, cur_state_dict, pre_state_dict, prefix="", load_name=None):
        def _get_params(key):
            key = prefix + key
            if key in pre_state_dict:
                return pre_state_dict[key]
            return None

        for k in cur_state_dict.keys():
            if load_name is not None:
                if load_name not in k:
                    continue
            v = _get_params(k)
            try:
                if v is None:
                    # print('parameter {} not found'.format(k))
                    continue
                cur_state_dict[k].copy_(v)
            except:
                # print('copy param {} failed'.format(k))
                continue

    def create_model(self):
        # set up parameters
        self.n_param = sum(self.param_dict.values())
        self.n_detail = 128

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param)  # .to(self.device)
        # self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)

        # load model
        model_path = self.pretrained_modelpath
        if os.path.exists(model_path):
            print(f"trained model found. load {model_path}")
            checkpoint = torch.load(model_path, weights_only=True)
            self.checkpoint = checkpoint
            self.copy_state_dict(self.E_flame.state_dict(), checkpoint["E_flame"])
            # util.copy_state_dict(self.E_detail.state_dict(), checkpoint["E_detail"])
        else:
            print(f"please check model path: {model_path}")
            # exit()

        self.E_flame.eval()
        # self.E_detail.eval()

    def decompose_code(self, code, num_dict):
        """Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        """
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == "light":
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    @torch.no_grad()
    def encode(self, images, use_detail=False):
        parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        # codedict["images"] = images
        if use_detail:
            detailcode = self.E_detail(images)
            codedict["detail"] = detailcode

        # if self.cfg.model.jaw_type == "euler":
        #     posecode = codedict["pose"]
        #     euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
        #     posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
        #     codedict["pose"] = posecode
        #     codedict["euler_jaw_pose"] = euler_jaw_pose

        return codedict["exp"]

    def forward(self, images):
        codedict = self.encode(images)
        return codedict

    def model_dict(self):
        return {
            "E_flame": self.E_flame.state_dict(),
            # "E_detail": self.E_detail.state_dict(),
        }
