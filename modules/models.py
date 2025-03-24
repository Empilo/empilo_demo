import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as tf
import math
import numpy as np
import os
from time import time
import imageio
from collections import defaultdict

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from modules.blocks import (
    DownBlock2d,
    ResBottleneck,
    ResBlock2d,
    UpBlock2d,
    SPADEResnetBlock,
    SPADEResnetBlock_cached,
    DownBlock_Disc,
)
from modules.utils import eval_sh, cam_world_matrix_transform, bs_normalize, deca_normalize
# from modules.loss import Vgg19, ParseNet
from modules.deca import DECA
from modules.renderer import RayMarcher


class FaceModel:
    def __init__(self, precision="fp32"):
        self.precision = precision
        base_options = python.BaseOptions(model_asset_path="pretrained/face_landmarker_v2_with_blendshapes.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.deca = DECA()
        if self.precision == "fp16":
            self.deca.half()
        self.deca.to(device="cuda")

    def detect(self, frame, time_log=None):
        B = frame.shape[0]
        vectors = []
        matrices = []
        for i in range(B):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame[i], dtype="uint8"))
            detection_result = self.detector.detect(mp_image)
            scores = [item.score for item in detection_result.face_blendshapes[0]]
            vectors.append(bs_normalize(scores))
            matrices.append(detection_result.facial_transformation_matrixes)

        vectors_np = np.array(vectors)
        matrices_np = np.array(matrices)

        # bs_vector = torch.tensor(vectors_np, dtype=torch.float32, requires_grad=False).to(device="cuda")
            
        frame_tensor = (frame / 255.0).permute(0, 3, 1, 2).to(device="cuda")
        
        if time_log is not None:
            torch.cuda.synchronize()
            t0 = time()
            
            bs_vector = deca_normalize(self.deca(F.interpolate(frame_tensor, size=(224, 224), mode="bilinear")))
            
            torch.cuda.synchronize()
            t1 = time()
            print(f"Encoding time: {(t1 - t0) * 1000:.2f} ms", file=time_log)
            
        else:
            bs_vector = deca_normalize(self.deca(F.interpolate(frame_tensor, size=(224, 224), mode="bilinear")))

        tf_matrix = torch.tensor(matrices_np, dtype=bs_vector.dtype, requires_grad=False).to(device="cuda")

        tf = cam_world_matrix_transform(tf_matrix.squeeze(1))  # B 4 4
        return bs_vector, tf


class Profiles(nn.Module):
    def __init__(self, P, H, W, channel=52, scale_factor=1):
        super().__init__()
        self.profile = nn.Parameter(torch.randn(P, channel, H // scale_factor, W // scale_factor, device="cuda"))


class Blender(nn.Module):
    def __init__(self, H, W, num_blendshapes=52, scale_factor=1):
        super().__init__()
        self.id_tensor = nn.Parameter(
            torch.randn(1, num_blendshapes, H // scale_factor, W // scale_factor, device="cuda") * 0.1
        )

    def forward(self, bs_vector, precision="fp32"):
        assert torch.is_tensor(bs_vector) == True, "BS vector should be torch tensor."
        B = bs_vector.shape[0]

        assert self.id_tensor.shape[1] == bs_vector.shape[1], "BS vector size & ID tensor channel mismatch."
        
        batched_id_tensor = self.id_tensor.expand(B, -1, -1, -1)

        blended = batched_id_tensor * bs_vector.view(B, -1, 1, 1)
        return blended


class Decoder(nn.Module):
    def __init__(self, num_resblocks=0, num_layers=4, max_features=1024, num_blendshapes=52, num_basis=16, superimpose=4):
        super().__init__()

        self.dec_blocks = nn.Sequential()

        for i in range(num_layers):
            fin = min(2 ** (i + 6), max_features) if i != 0 else num_blendshapes
            fout = min(2 ** (i + 7), max_features)
            self.dec_blocks.add_module(
                f"DownBlock{i}", DownBlock2d(in_features=fin, out_features=fout, kernel_size=3, padding=1)
            )

        for i in range(num_resblocks):
            self.dec_blocks.add_module(f"ResBlock{i}", ResBottleneck(in_features=fout, stride=1))

        for i in range(num_layers + 1):
            fin = min((2 ** (6 + num_layers - i)), max_features)

            fout = superimpose * num_basis if i == num_layers else min((2 ** (5 + num_layers - i)), max_features)

            if i < num_layers:
                self.dec_blocks.add_module(
                    f"UpBlock{i}", UpBlock2d(in_features=fin, out_features=fout, kernel_size=3, padding=1)
                )
            else:
                self.dec_blocks.add_module(
                    f"UpBlock{i}", UpBlock2d(in_features=fin, out_features=fout, kernel_size=3, padding=1, last_block=True)
                )

    def forward(self, x):
        out = self.dec_blocks(x)
        return out


class SPADEDecoder(nn.Module):
    def __init__(
        self,
        num_layers=4,
        max_features=1024,
        fhidden=128,
        num_blocksperscale=1,
        num_blocks=2,
        num_blendshapes=52,
        num_basis=16,
        superimpose=1,
    ):
        super().__init__()

        self.num_bps = num_blocksperscale
        # self.num_scale = int(math.log2(scale_factor))
        self.num_blocks = num_blocks
        self.decoder_front = nn.ModuleList()
        self.decoder_rear = Decoder(
            num_resblocks=0,
            max_features=max_features,
            num_layers=num_layers,
            num_blendshapes=num_blendshapes,
            num_basis=num_basis,
            superimpose=superimpose,
        )
        self.up = nn.Upsample(scale_factor=2)

        fin = fout = num_blendshapes
        for i in range(self.num_blocks):
            self.decoder_front.append(SPADEResnetBlock(fin, fout, fhidden))

    def forward(self, tensor, style):

        for i in range(self.num_blocks):
            tensor = self.decoder_front[i](tensor, style)
        tensor = self.decoder_rear(tensor)

        return tensor


# cached version
class SPADEDecoder2(nn.Module):
    def __init__(
        self,
        style,
        num_layers=4,
        max_features=1024,
        fhidden=128,
        num_blocksperscale=1,
        num_blendshapes=52,
        num_blocks=2,
        num_basis=16,
        superimpose=1,
    ):
        super().__init__()

        self.num_bps = num_blocksperscale
        # self.num_scale = int(math.log2(scale_factor))
        self.num_blocks = num_blocks
        self.decoder_front = nn.ModuleList()
        self.decoder_rear = Decoder(
            num_resblocks=0,
            max_features=max_features,
            num_layers=num_layers,
            num_blendshapes=num_blendshapes,
            num_basis=num_basis,
            superimpose=superimpose,
        )
        self.up = nn.Upsample(scale_factor=2)

        fin = fout = num_blendshapes
        for i in range(self.num_blocks):
            self.decoder_front.append(SPADEResnetBlock_cached(style, fin, fout, fhidden))

    def forward(self, tensor):

        for i in range(self.num_blocks):
            tensor = self.decoder_front[i](tensor)
        tensor = self.decoder_rear(tensor)

        return tensor


class Generator(nn.Module):
    def __init__(self, blender, decoder, discriminator, config, precision="fp32"):
        super().__init__()
        self.precision = precision
        # self.loss_weights = config["train_params"]["loss_weights"]
        
        self.facemodel = FaceModel(precision=self.precision)
        self.blender = blender
        self.decoder = decoder
        self.discriminator = discriminator

        self.renderer = RayMarcher(batch=config["test_params"]["batch_size"], 
                                   layer=config["model_params"]["superimpose"], 
                                   precision=self.precision)

        # self.vgg = Vgg19()
        # if torch.cuda.is_available():
        #     self.vgg = self.vgg.cuda()

        # self.parsenet = ParseNet()
        # if torch.cuda.is_available():
        #     self.parsenet = self.parsenet.cuda()

    def reconstruct(self, x, x_inj, add_noise=True, time_log=None):
        bs_vector, tf_matrix = self.facemodel.detect(x, time_log)
        
        tf_matrix[..., 2, 3] = (tf_matrix[:, 2, 3] - 25.0) * 2.0
        inv_tf = tf_matrix[..., :3, :]
        inv_tf[:, :, 3] = torch.nn.functional.normalize(inv_tf[:, :, 3], dim=1)

        if add_noise:
            bs_vector += (torch.randn(bs_vector.size(), device="cuda") - 0.5) * 0.1

        gt = x.permute(0, 3, 1, 2).to(device="cuda")  # B H W 3 -> B 3 H W
        style = x_inj.permute(0, 3, 1, 2).to(device="cuda")

        gt = gt / 127.5 - 1.0
        style = style / 127.5 - 1.0

        if time_log is not None:
            torch.cuda.synchronize()
            t0 = time()
            
            exp_tensor = self.blender(bs_vector, self.precision)
            sh_tensor = self.decoder(exp_tensor, style)
            
            torch.cuda.synchronize()
            t1 = time()
            
            predicted, features = self.renderer(sh_tensor, inv_tf)
            
            torch.cuda.synchronize()
            t2 = time()
            
            print(f"Total frame generation time: {(t2 - t0) * 1000:.2f} ms", file=time_log)
            print(f"↳ Re-rendering time: {(t2 - t1) * 1000:.2f} ms ({(t2 - t1)/(t2 - t0) * 100:.1f}% of total)\n", file=time_log)        
        
        else:
            exp_tensor = self.blender(bs_vector)
            sh_tensor = self.decoder(exp_tensor, style)
            predicted, features = self.renderer(sh_tensor, inv_tf)

        c = tf_matrix.flatten(start_dim=1)[:, :12]
        return predicted, gt, style, c

    def reconstruct_cross(self, x, pose, inj, add_noise=False, time_log=None):
        bs_vector, _ = self.facemodel.detect(x, time_log)
        _, tf_matrix = self.facemodel.detect(pose)
        
        tf_matrix[..., 2, 3] = (tf_matrix[:, 2, 3] - 25.0) * 2.0
        inv_tf = tf_matrix[..., :3, :]
        inv_tf[:, :, 3] = torch.nn.functional.normalize(inv_tf[:, :, 3], dim=1)

        if add_noise:
            bs_vector += (torch.randn(bs_vector.size(), device="cuda") - 0.5) * 0.05

        gt = x.permute(0, 3, 1, 2).to(device="cuda")  # B H W 3 -> B 3 H W
        style = inj.permute(0, 3, 1, 2).to(device="cuda")
        angle = pose.permute(0, 3, 1, 2).to(device="cuda")

        gt = gt / 127.5 - 1.0
        style = style / 127.5 - 1.0
        angle = angle / 127.5 - 1.0

        if time_log is not None:
            torch.cuda.synchronize()
            t0 = time()
            
            exp_tensor = self.blender(bs_vector, self.precision)
            sh_tensor = self.decoder(exp_tensor, style)
            
            torch.cuda.synchronize()
            t1 = time()
            
            cross_predicted, cross_features = self.renderer(sh_tensor, inv_tf)
            
            torch.cuda.synchronize()
            t2 = time()
            
            print(f"Total frame generation time: {(t2 - t0) * 1000:.2f} ms", file=time_log)
            print(f"↳ Re-rendering time: {(t2 - t1) * 1000:.2f} ms ({(t2 - t1)/(t2 - t0) * 100:.1f}% of total)\n", file=time_log)    
            
        else:
            exp_tensor = self.blender(bs_vector)
            sh_tensor = self.decoder(exp_tensor, style)
            cross_predicted, cross_features = self.renderer(sh_tensor, inv_tf)

        
        c_cross = tf_matrix.flatten(start_dim=1)[:, :12]
        return cross_predicted, gt, style, angle, c_cross

    def loss(self, predicted, gt, cross_predicted=None, gt_pose=None, c=None, c_cross=None, phase="phase_1"):
        losses_gen = {}
        losses_disc = {}
        phase_weights = self.loss_weights[phase]

        # L1 Loss
        if "l1" in phase_weights:
            l1_loss = nn.L1Loss()
            loss = l1_loss(predicted, gt)
            losses_gen["l1"] = phase_weights["l1"] * loss

        # L2 Loss
        if "l2" in phase_weights:
            l2_loss = nn.MSELoss()
            loss = l2_loss(predicted, gt)
            losses_gen["l2"] = phase_weights["l2"] * loss

        # Perceptual Loss
        if self.vgg is not None and "perceptual" in phase_weights:
            predicted_01 = (predicted + 1.0) / 2.0
            gt_01 = (gt + 1.0) / 2.0
            gt_vgg = self.vgg(gt_01)
            predicted_vgg = self.vgg(predicted_01)

            perceptual_loss = 0
            for i, weight in enumerate(phase_weights["perceptual"]):
                value = torch.abs(predicted_vgg[i] - gt_vgg[i].detach()).mean()
                perceptual_loss += weight * value
            losses_gen["perceptual"] = perceptual_loss
            
        # and another ...

        return losses_gen, losses_disc