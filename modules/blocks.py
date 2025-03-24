from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


class ResBottleneck(nn.Module):
    def __init__(self, in_features, stride):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(
            in_channels=in_features // 4, out_channels=in_features // 4, kernel_size=3, padding=1, stride=stride
        )
        self.conv3 = nn.Conv2d(in_channels=in_features // 4, out_channels=in_features, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_features // 4, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features // 4, affine=True)
        self.norm3 = nn.BatchNorm2d(in_features, affine=True)

        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=stride)
            self.norm4 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        out += x
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, last_block=False):
        super(UpBlock2d, self).__init__()

        self.last_block = last_block
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        if not self.last_block:
            out = F.relu(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std**2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, :: self.int_inv_scale, :: self.int_inv_scale]

        return out


class SPADE(nn.Module):
    def __init__(self, nc_in, nc_hidden, nc_style=3, normalize=True):
        super().__init__()

        self.normalize = normalize
        self.param_free_norm = nn.InstanceNorm2d(nc_in, affine=False)

        self.mlp_shared = nn.Sequential(nn.Conv2d(nc_style, nc_hidden, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nc_hidden, nc_in, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nc_hidden, nc_in, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        if self.normalize:
            normalized = self.param_free_norm(x)
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode="bilinear", antialias=True)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADE2(nn.Module):
    def __init__(self, segmap, nc_in, nc_hidden, resize, nc_style=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(nc_in, affine=False)

        self.mlp_shared = nn.Sequential(nn.Conv2d(nc_style, nc_hidden, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nc_hidden, nc_in, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nc_hidden, nc_in, kernel_size=3, padding=1)

        # half precision (!)
        # self.mlp_shared.to(device="cuda:0").half()
        # self.mlp_gamma.to(device="cuda:0").half()
        # self.mlp_beta.to(device="cuda:0").half()

        # if resize:
        #     segmap = F.interpolate(segmap, scale_factor=0.5, mode="bilinear", antialias=True)

        # actv = self.mlp_shared(segmap)
        # self.gamma = self.mlp_gamma(actv)
        # self.beta = self.mlp_beta(actv)

        # onnx
        self.gamma = torch.randn(1, nc_in, 256, 256)
        self.beta = torch.randn(1, nc_in, 256, 256)

    def forward(self, x):
        # normalized = self.param_free_norm(x)
        normalized = x
        out = normalized * (1 + self.gamma) + self.beta
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden, nc_style=3, spectral=True):
        super().__init__()

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.convs = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spectral:
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            if self.learned_shortcut:
                self.convs = spectral_norm(self.convs)

        # define normalization layers
        self.norm0 = SPADE(fin, fhidden, nc_style)
        self.norm1 = SPADE(fmiddle, fhidden, nc_style)
        if self.learned_shortcut:
            self.norms = SPADE(fin, fhidden, nc_style)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            xs = self.convs(self.norms(x, seg))
        else:
            xs = x
        return xs

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x, seg):
        xs = self.shortcut(x, seg)
        dx = self.conv0(self.actvn(self.norm0(x, seg)))
        dx = self.conv1(self.actvn(self.norm1(dx, seg)))
        out = xs + dx
        return out


class SPADEResnetBlock_cached(nn.Module):
    def __init__(self, seg, fin, fout, fhidden, resize=False, nc_style=3, spectral=False):  # True
        super().__init__()

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.convs = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spectral:
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            if self.learned_shortcut:
                self.convs = spectral_norm(self.convs)

        # define normalization layers
        self.norm0 = SPADE2(seg, fin, fhidden, resize)
        self.norm1 = SPADE2(seg, fmiddle, fhidden, resize)
        if self.learned_shortcut:
            self.norms = SPADE2(seg, fin, fhidden, resize)

    # def shortcut(self, x, seg):
    #     if self.learned_shortcut:
    #         xs = self.convs(self.norms(x, seg))
    #     else:
    #         xs = x
    #     return xs

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x):
        # xs = self.convs(self.norms(x))
        xs = x
        dx = self.conv0(self.actvn(self.norm0(x)))
        dx = self.conv1(self.actvn(self.norm1(dx)))
        out = xs + dx
        return out


class DownBlock_Disc(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock_Disc, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out
