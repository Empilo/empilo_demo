import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import yaml
from argparse import ArgumentParser
from shutil import copy
import datetime
import torch

from modules.models import Blender, SPADEDecoder
# from modules.discriminator import DualDiscriminator
# from train import train
from test import test
from datasetter import FrameDataset


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/obama.yaml", help="path to config")
    parser.add_argument("--log_dir", default="log", help="path to log into")
    parser.add_argument("--ckp", default=None, help="path to checkpoint to restore")
    parser.add_argument("--mode", choices=["train", "test_recon", "test_style", "test_pose", "test_style_n_pose"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
        
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.ckp is not None:
        log_dir = os.path.join(*os.path.split(opt.ckp)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config.split("/")[-1].split(".")[0]))
        local_time = datetime.datetime.now()
        log_dir += "_" + local_time.strftime("%y-%m-%d_%H-%M-%S")

    blender = Blender(
        config["dataset_params"]["H"],
        config["dataset_params"]["W"],
        config["model_params"]["num_blendshapes"],
        config["model_params"]["scale_factor"],
    )

    decoder = SPADEDecoder(
        config["model_params"]["decoder"]["num_layers"],
        config["model_params"]["decoder"]["max_features"],
        config["model_params"]["decoder"]["fhidden"],
        config["model_params"]["decoder"]["num_blocksperscale"],
        config["model_params"]["decoder"]["num_blocks"],
        config["model_params"]["num_blendshapes"],
        config["model_params"]["num_basis"],
        config["model_params"]["superimpose"],
    )

    discriminator = (
        DualDiscriminator(
            c_dim=12,
            img_resolution=config["dataset_params"]["H"],
            img_channels=3,
        ) if opt.mode == "train"
        else None
    )

    if torch.cuda.is_available():
        print(" cuda is available.")
        blender.to("cuda")
        decoder.to("cuda")
        if discriminator is not None:
            discriminator.to("cuda") 
        print()

    if opt.verbose:
        # print(styler)
        print(blender.id_tensor.shape)
        print(decoder)
        if discriminator is not None:
            print(discriminator.extra_repr())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if opt.mode == "train":
        print(" ========================= Training Mode ====================================")
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)
        train_set = FrameDataset(directories=config["dataset_params"]["train_directory"])
        val_set = FrameDataset(directories=config["dataset_params"]["val_directory"])
        train(config, blender, decoder, discriminator, opt.ckp, log_dir, train_set, val_set, device_ids)

    if opt.mode.startswith("test"):
        print(" ========================= Inference Mode ====================================")
        test_set = FrameDataset(directories=config["dataset_params"]["test_directory"])
        test(config, blender, decoder, opt.ckp, log_dir, test_set, opt.mode, opt.precision)
