import tqdm
from logger import Logger
import imageio

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast

from modules.models import Generator

def test(config, blender, decoder, checkpoint, log_dir, test_set, mode, precision):
    
    if checkpoint is not None:
        epoch = Logger.load_cpk(
            checkpoint_path=checkpoint,
            models={
                "blender": blender,
                "decoder": decoder,
            },
        )
        print(f" Trained through epoch {epoch}.")
    else:
        print(f" No checkpoint to evaluate.")
    
    if precision == "fp16":
        blender = blender.half()
        decoder = decoder.half()
    
    blender.eval()
    decoder.eval()

    generator = Generator(blender, decoder, None, config, precision)

    with Logger(log_dir=log_dir, mode=mode, trained_epoch=epoch) as logger:
        
        dataloader = DataLoader(test_set, batch_size=config["test_params"]["batch_size"], num_workers=4, shuffle=False)

        for i, (x, x_inj, y, y_inj, nums, idxs) in enumerate(tqdm.tqdm(dataloader)):
            
            if precision == "fp16":
                x = x.half()
                x_inj = x_inj.half()
                y = y.half()
                y_inj = y_inj.half()
                z = z.half()

            print(f"--- Frame {str(i).zfill(3)} ---", file=logger.log_file)
            if mode[5:] == "recon":
                predicted, gt, _, _ = generator.reconstruct(x, x_inj, add_noise=False, time_log=logger.log_file)
                logger.visualize_rec([gt, predicted], i)
                
            if mode[5:] == "style":
                predicted, gt, style, _ = generator.reconstruct(x, y_inj, add_noise=False, time_log=logger.log_file)
                logger.visualize_rec([gt, predicted, style], i)
            
            if mode[5:] == "pose":
                predicted, gt, _, angle, _ = generator.reconstruct_cross(x, y, x_inj, add_noise=False, time_log=logger.log_file)
                logger.visualize_rec([gt, predicted, angle], i)
            