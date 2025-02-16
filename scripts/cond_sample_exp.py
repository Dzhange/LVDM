import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import trange
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import torch_to_np, str2bool
from lvdm.utils.dist_utils import setup_dist, gather_data
from scripts.sample_utils import (
    load_model,
    save_args,
    make_model_input_shape,
    sample_batch,
    save_results,
)
# Example dataset for demonstration:
from lvdm.data.pusht_video_dataset import PushTVideoDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

from IPython import embed

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="model checkpoint path")
    parser.add_argument("--first_stage_ckpt", type=str, help="first stage model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
    # DDP / GPU
    parser.add_argument("--ddp", action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    # Sampling
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_type", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--show_denoising_progress", action='store_true', default=False)
    # Saving
    parser.add_argument("--save_mp4", type=str2bool, default=True)
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False)
    parser.add_argument("--save_npz", action='store_true', default=False)
    parser.add_argument("--save_jpg", action='store_true', default=False)
    parser.add_argument("--save_fps", type=int, default=8)
    return parser

@torch.no_grad()
def sample_conditioned(
    model, 
    noise_shape,     # e.g. [B, C, T, H, W]
    condition,       # shape [3, H, W] or [B, 3, H, W], your single image
    n_iters,
    ddp=False,
    **kwargs
):
    """
    This function takes a single-frame condition and applies the same 
    logic as `LatentDiffusion.log_images()` to encode the condition 
    properly before sampling.
    """
    # 1) Encode the raw condition if the model has a learnable cond_stage_model
    #    shape must be [B, C, H, W] if we call get_learned_conditioning directly.
    #    If you have only 1 frame in 'condition', expand batch dimension:
    if condition.dim() == 3:
        condition = condition.unsqueeze(0)  # shape [1, 3, H, W]
    condition = condition.to(model.device)
    
    # This call uses the same internal logic as in LatentDiffusion:
    encoded_cond = model.get_learned_conditioning(condition)
    # If your model expects [B, c, T, h, w], replicate along time dimension:
    # noise_shape = [B, model.channels, T, H, W], so let's replicate T times
    time_dim = noise_shape[2]
    # encoded_cond could already have multiple channels, etc.
    # Expand a temporal dim if it doesn't exist:
    if encoded_cond.dim() == 4:
        encoded_cond = encoded_cond.unsqueeze(2)  # [B, c, 1, h, w]
    encoded_cond = encoded_cond.repeat(1, 1, time_dim, 1, 1)

    # We'll sample in loops, each loop sampling `noise_shape[0]` (batch) at once
    all_videos = []
    for _ in trange(n_iters, desc="Sampling Batches (conditional)"):
        # 2) replicate across the batch dimension
        c_b = encoded_cond.repeat(noise_shape[0], 1, 1, 1, 1)
        cond_dict = {
            "first_frame": c_b
        }
        
        # 3) Actually run the forward diffusion sampling
        samples = sample_batch(model, noise_shape, condition=cond_dict, **kwargs)
        
        # 4) Decode the latent samples into pixel space
        videos = model.decode_first_stage(samples)
        
        # 5) Gather across all GPUs if using DDP
        if ddp:
            data_list = gather_data(videos, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(videos))

    all_videos = np.concatenate(all_videos, axis=0)
    return all_videos

def save_images(samples: torch.Tensor, out_dir: str, image_idx: int) -> None:
    """
    Save images arranged in a single row
    """

    sample = samples[0]

    sample = sample.permute(1, 0, 2, 3)    
    n = sample.shape[0]    
    grid = make_grid(sample, nrow=n, padding=0)
    
    filename = os.path.join(out_dir, f"image__{image_idx:04d}.png")
    save_image(grid, filename)

def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    save_args(opt.save_dir, opt)
    
    if opt.ddp:
        setup_dist(opt.local_rank)
        opt.n_samples = math.ceil(opt.n_samples / dist.get_world_size())
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.gpu_id}"
    
    if opt.seed is not None:
        # Offset seeds by local_rank if in DDP
        seed = opt.local_rank + opt.seed if opt.ddp else opt.seed
        seed_everything(seed)

    # Load config
    config = OmegaConf.load(opt.config_path)
    # Patch in any missing paths
    config.model.params.first_stage_config.params.ckpt_path = opt.first_stage_ckpt
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)

    # Load the latent-diffusion model
    model, _, _ = load_model(config, opt.ckpt_path)

    # Create sampler if needed
    ddim_sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None

    # --------------------------------------------------------
    # Initialize the dataset.
    dataset = PushTVideoDataset(
        data_root='/home/ubuntu/Robot/LVDM/datasets/push_t',
        resolution=64,
        video_length=16,
        subset_split='test',
        spatial_transform='center_crop_resize',
        clip_step=1,
        include_first_frame=True
    )

    # Create a DataLoader for the dataset.
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        # shuffle=False,
        worker_init_fn=None,
        collate_fn=None,
    )

    # Iterate over the entire dataset.
    cnt = 0
    for batch_idx, batch in enumerate(loader):
        # if batch_idx % 64 != 0:

        #     continue
        # if cnt >= 15:            
        #     break
        # Generate logged images from the current batch.
        log = model.log_images(batch)
        print(f"After logging image for batch {batch_idx}!")

        # Optionally, drop into an interactive shell for debugging.
        # embed()

        # Extract the samples from the log and save the images.
        samples = log['samples']
        save_images(samples, opt.save_dir, batch_idx)

    # Clean up the process group if using Distributed Data Parallel.
    if opt.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
