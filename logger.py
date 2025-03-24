import numpy as np
import torch
import torch.nn.functional as F
import imageio
import os
import time
import datetime

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, mode, checkpoint_freq=20, zfill_num=4, trained_epoch=0):
        self.loss_list = []
        self.cpk_dir = log_dir
        self.epoch = trained_epoch
        self.visualizations_dir = os.path.join(log_dir, f"{mode}-vis")
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.zfill_num = zfill_num
        self.mode = mode
        self.log_file_name = mode + "_log"
        self.log_file = open(os.path.join(log_dir, self.log_file_name + ".txt"), "a")
        self.visualizer = Visualizer()
        self.checkpoint_freq = checkpoint_freq
        self.best_loss = float("inf")
        self.names = None
        local_time = datetime.datetime.now()
        self.time = local_time.strftime("%y-%m-%d_%H-%M-%S")

    def log_scores(self, loss_names, img_idx="", latency=None):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s- %.4f" % (name, value) for name, value in zip(loss_names, loss_mean)])

        if latency is not None:
            loss_string = loss_string + f" ; {round((latency*1000.0), 2)}ms"

        if self.mode.startswith("test"):
            loss_string = f"{str(img_idx).zfill(self.zfill_num)}.png; " + loss_string

        loss_string = "epoch " + str(self.epoch).zfill(self.zfill_num) + "; " + loss_string

        print(loss_string, file=self.log_file)
        print(loss_string)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, images_list, img_idx="sample"):
        image = self.visualizer.visualize(images_list)
        if self.mode.startswith("train"):
            file_name = os.path.join(self.visualizations_dir, f"{str(self.epoch)}-{self.mode}.png")
        if self.mode.startswith("test"):
            test_dir = os.path.join(self.visualizations_dir, f"epoch{self.epoch}")
            if not (os.path.exists(test_dir)):
                os.makedirs(test_dir)
            file_name = os.path.join(test_dir, f"{str(img_idx)}.png")
        imageio.imsave(file_name, image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items() if v is not None}
        cpk["epoch"] = self.epoch + 1
        cpk_path = os.path.join(self.cpk_dir, "%s-checkpoint.pth.tar" % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, models=None, optimizers=None, schedulers=None):
        checkpoint = torch.load(checkpoint_path)
        # Load models
        if models is not None:
            for name, model in models.items():
                if name in checkpoint:
                    model.load_state_dict(checkpoint[name])

        # Load optimizers
        if optimizers is not None:
            for name, optimizer in optimizers.items():
                optimizer_key = f"optimizer_{name}"
                if optimizer_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[optimizer_key])

        # Load schedulers
        if schedulers is not None:
            for name, scheduler in schedulers.items():
                scheduler_key = f"scheduler_{name}"
                if scheduler_key in checkpoint:
                    scheduler.load_state_dict(checkpoint[scheduler_key])

        return checkpoint["epoch"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if "models" in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, images_list):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(images_list)
        self.names = None

    def log_test(self):
        print()


class Visualizer:
    def __init__(self, draw_border=False):
        self.draw_border = draw_border

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, images_list):
        # Process each tensor in the list
        processed_images = [
            np.transpose(127.5 * (img.data.cpu() + 1), [0, 2, 3, 1]) if img is not None else None for img in images_list
        ]

        # Filter out None values
        processed_images = [img for img in processed_images if img is not None]

        # Create image grid
        image = self.create_image_grid(*processed_images)
        return image.astype(np.uint8)
