# ----------------------------
# File: train_inverse_dynamics.py
# ----------------------------

import argparse
import yaml
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
import sys

# Import your dataset (already implemented elsewhere)
# from my_dataset import PushTVideoDataset
from lvdm.models.idm import InverseDynamicsModel
from lvdm.data.pusht_video_dataset import PushTVideoDataset
###############################################################################
#                        Visualization Helper
###############################################################################
def visualize_actions(img, pred_action, gt_action, step, viz_dir):
    """
    Overlays predicted and ground-truth 2D action vectors on the given image.
    
    Args:
        img:         (3, H, W) tensor in [-1, 1].
        pred_action: (2,) predicted action [dx, dy].
        gt_action:   (2,) ground truth action [dx, dy].
        step:        index (int) for naming the output file.
        viz_dir:     directory to save the visualization images.
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert from [-1,1] to [0,1] for plotting
    img_np = (img.cpu().numpy() * 0.5) + 0.5  # shape: (3, H, W)
    img_np = np.clip(img_np, 0.0, 1.0)
    
    # Reorder for matplotlib: (H, W, 3)
    img_np = np.transpose(img_np, (1, 2, 0))
    
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
        -pred_action[1].item() * 10, # negative y to match image coords
        width=1.0, color='red', length_includes_head=True,
        head_width=5, head_length=8, alpha=0.7, label='Predicted'
    )
    
    # Ground-truth action (blue)
    ax.arrow(
        center_x, center_y,
        gt_action[0].item() * 10,
        -gt_action[1].item() * 10,
        width=1.0, color='blue', length_includes_head=True,
        head_width=5, head_length=8, alpha=0.7, label='Ground Truth'
    )
    
    ax.legend()
    ax.set_title(f"Pred vs GT Actions @ step {step}")
    ax.axis("off")
    
    out_path = os.path.join(viz_dir, f"viz_step_{step}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


###############################################################################
#                       Main Training Function
###############################################################################
def train_inverse_dynamics(config):
    """
    Train the inverse-dynamics model using a config dict loaded from YAML.
    Includes:
     - Logging
     - Visualization of predicted vs. GT actions
     - Checkpoint saving on keyboard interrupt
    """
    # 0) Setup logging
    logger = logging.getLogger(__name__)
    logger.info("Starting training with config:")
    logger.info(config)

    # 1) Create result_dir if doesn't exist
    result_dir = config["train"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    viz_dir = os.path.join(result_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # 2) Load dataset
    #    (assuming your dataset is in my_dataset.PushTVideoDataset)
    
    dataset = PushTVideoDataset(
        data_root=config["dataset"]["data_root"],
        resolution=config["dataset"]["resolution"],
        video_length=config["dataset"]["video_length"],
        subset_split=config["dataset"].get("subset_split", "test"),
        spatial_transform=config["dataset"].get("spatial_transform", "center_crop_resize"),
        clip_step=config["dataset"].get("clip_step", 1)
    )
    loader = data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"].get("num_workers", 2),
        drop_last=True
    )

    # 3) Initialize model & optimizer
    device = config["train"].get("device", "cuda")
    action_dim = config["model"]["action_dim"]
    model = InverseDynamicsModel(action_dim=action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # Choose loss
    loss_type = config["train"].get("loss_type", "MSE")
    if loss_type == "MSE":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Only MSE is supported in this example.")
    
    # 4) Training setup
    epochs = config["train"]["epochs"]
    log_steps = config["train"].get("log_steps", 50)
    logger.info(f"Training for {epochs} epochs on {len(dataset)} samples.")
    
    model.train()
    global_step = 0

    # 5) Training loop with KeyboardInterrupt handling
    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            step_count = 0

            for batch_idx, batch in enumerate(loader):
                images = batch["image"].to(device)   # (B, 3, T, H, W)
                actions = batch["action"].to(device) # (B, T, action_dim)
                B, C, T, H, W = images.shape

                # Build consecutive frame pairs
                img_t_list, img_t1_list, action_list = [], [], []
                for t_idx in range(T - 1):
                    img_t_list.append(images[:, :, t_idx])
                    img_t1_list.append(images[:, :, t_idx + 1])
                    action_list.append(actions[:, t_idx])

                img_t_batch  = torch.cat(img_t_list, dim=0)   # (B*(T-1), 3, H, W)
                img_t1_batch = torch.cat(img_t1_list, dim=0)  # (B*(T-1), 3, H, W)
                action_batch = torch.cat(action_list, dim=0)  # (B*(T-1), action_dim)

                # Forward
                pred_actions = model(img_t_batch, img_t1_batch)
                loss = criterion(pred_actions, action_batch)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step_count += 1
                global_step += 1

                # Logging & Visualization
                if global_step % log_steps == 0:
                    avg_loss = epoch_loss / step_count
                    logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}")

                    # Visualization (draw for the first sample in the batch)
                    # Only if we have at least 1 sample
                    if img_t_batch.shape[0] > 0:
                        sample_img_t = img_t_batch[0].detach()
                        sample_pred  = pred_actions[0].detach()
                        sample_gt    = action_batch[0].detach()
                        
                        visualize_actions(
                            img=sample_img_t,
                            pred_action=sample_pred,
                            gt_action=sample_gt,
                            step=global_step,
                            viz_dir=viz_dir
                        )

            epoch_avg_loss = epoch_loss / max(step_count, 1)
            logger.info(f"==> [Epoch {epoch}/{epochs}] Avg Loss: {epoch_avg_loss:.4f}")

        # Finished all epochs: save final checkpoint
        final_ckpt_path = os.path.join(result_dir, f"inverse_dynamics_step_{global_step}.pt")
        torch.save(model.state_dict(), final_ckpt_path)
        logger.info(f"Training complete! Final checkpoint: {final_ckpt_path}")

    except KeyboardInterrupt:
        # Handle Ctrl+C -> save an intermediate checkpoint
        logger.info("KeyboardInterrupt caught. Saving checkpoint before exit...")
        interrupt_ckpt_path = os.path.join(result_dir, f"inverse_dynamics_step_{global_step}.pt")
        torch.save(model.state_dict(), interrupt_ckpt_path)
        logger.info(f"Checkpoint saved to: {interrupt_ckpt_path}")
        sys.exit(0)

    return model


###############################################################################
#                              Main Entry Point
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Train Inverse Dynamics Model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()

    # 0) Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1) Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2) Train (includes try/except for KeyboardInterrupt in the function)
    model = train_inverse_dynamics(config)

    # (No further saving needed here because it's already done in train_inverse_dynamics)


if __name__ == "__main__":
    main()
