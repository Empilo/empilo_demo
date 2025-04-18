import numpy as np
import torch
from modules.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue

# from torch_utils import persistence
# from torch_utils.ops import upfirdn2d


class DualDiscriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=4,  # Use FP16 for the N highest resolutions.
        conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise=0,  # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
        has_superresolution=False,
        has_uv=False,
        has_seg=False,
    ):
        super().__init__()
        self.has_superresolution = has_superresolution
        self.has_uv = has_uv
        self.has_seg = has_seg
        if has_superresolution:
            img_channels *= 2
        if has_uv:
            img_channels += 3
        if has_seg:
            img_channels += 1

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2**i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs
        )
        self.disc_c_noise = disc_c_noise

    def forward(
        self,
        img,
        c,
        img_name="raw",
        seg_name="seg",
        **block_kwargs,
    ):

        if self.has_seg:
            raw = img[img_name]
            seg = img[seg_name]
            img = torch.cat([raw, seg], 1)  # image to be (B,C,H,W)

        else:
            img = img

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0:
                c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f"c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}"
