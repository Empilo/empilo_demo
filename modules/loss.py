import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms

import numpy as np
import math
from pretrained.parsing.model import BiSeNet


class Vgg19(nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = nn.Parameter(
            data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))), requires_grad=False
        )
        self.std = nn.Parameter(
            data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))), requires_grad=False
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ParseNet(nn.Module):

    def __init__(self, requires_grad=False):
        super(ParseNet, self).__init__()
        self.parsenet = BiSeNet(n_classes=19)
        self.parsenet.load_state_dict(torch.load("pretrained/79999_iter.pth", weights_only=True))
        self.parsenet.eval()

        # self.transform = transforms.Compose(
        #     [
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ]
        # )

        self.mean = nn.Parameter(
            data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))), requires_grad=False
        )
        self.std = nn.Parameter(
            data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))), requires_grad=False
        )

    def forward(self, x):
        out = (x - self.mean) / self.std  # B 3 H W
        cls_x = self.parsenet(out)[0].argmax(1).unsqueeze(1)  # B 1 H W

        mask0 = cls_x == 0
        mask1 = torch.logical_or(mask0, cls_x == 14)
        mask2 = torch.logical_or(mask1, cls_x == 15)
        mask3 = torch.logical_or(mask2, cls_x == 16)

        # out.masked_fill_(mask3, 0)

        out.masked_fill_(~mask3, 0)

        return out
