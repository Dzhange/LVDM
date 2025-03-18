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

# Import the IDM model for inference.
from lvdm.models.idm import InverseDynamicsModel

from IPython import embed

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="latent diffusion model checkpoint path")
    parser.add_argument("--first_stage_ckpt", type=str, help="first stage model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
    # IDM checkpoint (for inference on generated samples)
    parser.add_argument("--idm_ckpt", type=str, default=None, help="IDM model checkpoint path")
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

def visualize_actions(img, pred_action, gt_action, step, viz_dir):
    """
    Overlays predicted and ground-truth 2D action vectors on the given image.
    Args:
        img:         (3, H, W) tensor in [-1, 1].
        pred_action: (2,) predicted action [dx, dy].
        gt_action:   (2,) ground truth action [dx, dy] (here used as a dummy).
        step:        index (int) for naming the output file.
        viz_dir:     directory to save the visualization images.
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert from [-1,1] to [0,1] for plotting
    img_np = (img.cpu().numpy() * 0.5) + 0.5  # shape: (3, H, W)
    img_np = np.clip(img_np, 0.0, 1.0)
    # Reorder for matplotlib: (H, W, 3)
    img_np = np.transpose(img_np, (1, 2, 0))
    
    import matplotlib.pyplot as plt
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)
    
    H, W = img_np.shape[:2]
    center_x = W // 2
    center_y = H // 2
    
    # Predicted action (red)
    ax.arrow(
        center_x, center_y, 
        pred_action[0].item() * 10,   # scaled for visibility
        -pred_action[1].item() * 10,   # negative y to match image coords
        width=1.0, color='red', length_includes_head=True,
        head_width=5, head_length=8, alpha=0.7, label='Predicted'
    )
    
    # Dummy Ground-truth action (blue)
    ax.arrow(
        center_x, center_y,
        gt_action[0].item() * 10,
        -gt_action[1].item() * 10,
        width=1.0, color='blue', length_includes_head=True,
        head_width=5, head_length=8, alpha=0.7, label='Ground Truth'
    )
    
    ax.legend()
    ax.set_title(f"IDM Inference @ step {step}")
    ax.axis("off")
    
    out_path = os.path.join(viz_dir, f"viz_step_{step}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

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
    if condition.dim() == 3:
        condition = condition.unsqueeze(0)  # shape [1, 3, H, W]
    condition = condition.to(model.device)
    
    # This call uses the same internal logic as in LatentDiffusion:
    encoded_cond = model.get_learned_conditioning(condition)
    # If your model expects [B, c, T, h, w], replicate along time dimension:
    time_dim = noise_shape[2]
    if encoded_cond.dim() == 4:
        encoded_cond = encoded_cond.unsqueeze(2)  # [B, c, 1, h, w]
    encoded_cond = encoded_cond.repeat(1, 1, time_dim, 1, 1)

    all_videos = []
    for _ in trange(n_iters, desc="Sampling Batches (conditional)"):
        # replicate across the batch dimension
        c_b = encoded_cond.repeat(noise_shape[0], 1, 1, 1, 1)
        cond_dict = {
            "first_frame": c_b
        }
        
        # run the forward diffusion sampling
        samples = sample_batch(model, noise_shape, condition=cond_dict, **kwargs)
        # decode the latent samples into pixel space
        videos = model.decode_first_stage(samples)
        
        if ddp:
            data_list = gather_data(videos, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(videos))

    all_videos = np.concatenate(all_videos, axis=0)
    return all_videos

def save_images(samples: torch.Tensor, out_dir: str, image_idx: int) -> None:
    """
    Save images arranged in a single row.
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

    # Load config for the latent diffusion model.
    config = OmegaConf.load(opt.config_path)
    config.model.params.first_stage_config.params.ckpt_path = opt.first_stage_ckpt
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)

    # Load the latent diffusion model.
    model, _, _ = load_model(config, opt.ckpt_path)
    
    # Create the sampler if needed.
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
    print(len(dataset))
    # Create a DataLoader for the dataset.
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
    )

    # Create a directory for IDM visualizations.
    idm_viz_dir = os.path.join(opt.save_dir, "idm_viz")
    os.makedirs(idm_viz_dir, exist_ok=True)
    
    # Instantiate the IDM model.
    idm_model = InverseDynamicsModel(action_dim=2).to(model.device)
    if opt.idm_ckpt is not None:
        state_dict = torch.load(opt.idm_ckpt, map_location=model.device)
        idm_model.load_state_dict(state_dict)
    idm_model.eval()

    # Iterate over the entire dataset.
    cnt = 0
    for batch_idx, batch in enumerate(loader):
        # Generate logged images from the current batch.
        log = model.log_images(batch)
        print(f"After logging image for batch {batch_idx}!")
        # Extract the samples from the log.
        # Expected shape: [B, C, T, H, W]
        samples = log['samples']
        # Save the generated images (e.g., a grid from the first sample).
        save_images(samples, opt.save_dir, batch_idx)
        # ---------------------------
        # IDM Inference on Generated Samples
        # ---------------------------
        # For each generated video in the batch, we take the first two frames
        # as input to the IDM model.
        # Assume `samples` is a torch.Tensor.
        for i in range(samples.shape[0]):
            # Get a single video sample: shape [C, T, H, W]
            video = samples[i]
            # Extract frame 0 and frame 1.
            frame0 = video[:, 0, :, :].unsqueeze(0)  # shape [1, 3, H, W]
            frame1 = video[:, 1, :, :].unsqueeze(0)
            # Convert from [0,1] to [-1,1] if needed.
            frame0 = frame0 * 2 - 1
            frame1 = frame1 * 2 - 1
            with torch.no_grad():
                pred_action = idm_model(frame0, frame1)  # shape [1, 2]
            # Create a dummy ground-truth (e.g., zeros) for visualization.
            # dummy_gt = torch.zeros_like(pred_action)
            # embed()
            dummy_gt = batch['action'][0, i:i+1, :]
            # Visualize on the first frame.
            # We use the current batch index and sample index to name the output.
            viz_step = batch_idx * samples.shape[0] + i
            visualize_actions(frame0[0], pred_action[0], dummy_gt[0], viz_step, idm_viz_dir)
            print(f"Saved IDM visualization for sample {viz_step}")

    # Clean up the process group if using Distributed Data Parallel.
    if opt.ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
