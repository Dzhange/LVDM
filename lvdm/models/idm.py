# -------------------------
# File: inverse_dynamics_model.py
# -------------------------
import torch
import torch.nn as nn

class InverseDynamicsModel(nn.Module):
    """
    Simple inverse-dynamics model:
      - Expects two frames (img_t, img_t1) each of shape (B, 3, H, W).
      - Outputs an action vector of size `action_dim`.
    """
    def __init__(self, action_dim=2):
        super(InverseDynamicsModel, self).__init__()
        
        # Convolutional feature extractor
        # 6 input channels = 2 frames x 3 channels each
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Fully connected action predictor
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # e.g. 2D, 6D, etc.
        )

    def forward(self, img_t: torch.Tensor, img_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_t:  (B, 3, H, W) - Frame at time t
            img_t1: (B, 3, H, W) - Frame at time t+1
        
        Returns:
            actions: (B, action_dim)
        """
        # Concatenate along channel dimension: (B, 6, H, W)
        x = torch.cat([img_t, img_t1], dim=1)
        x = self.conv_layers(x)      # (B, 128, 8, 8)
        x = x.view(x.size(0), -1)    # Flatten -> (B, 128*8*8)
        actions = self.fc_layers(x)  # (B, action_dim)
        return actions
