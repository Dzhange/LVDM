from typing import Dict, Callable
import torch
import numpy as np
import copy
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from pathlib import Path

import sys
sys.path.append("/home/ubuntu/Robot/LVDM/lvdm/data/")
from replay_buffer import ReplayBuffer
from sampler import SequenceSampler, get_val_mask, downsample_mask

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class PushTVideoDataset:
    def __init__(self,
            data_root,
            resolution=96,
            video_length=16,
            subset_split='test',
            spatial_transform='center_crop_resize',
            clip_step=1,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            include_first_frame=False  # New flag
            ):
        
        super().__init__()
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.spatial_transform = spatial_transform
        self.clip_step = clip_step
        self.include_first_frame = include_first_frame  # Store the flag
        
        # Load data
        zarr_path = Path(data_root) / 'pusht_cchi_v7_replay.zarr'
        self.replay_buffer = ReplayBuffer.copy_from_path(
            str(zarr_path), keys=['img', 'state', 'action'])
        
        # Setup splits
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # Calculate required sequence length based on clip_step
        self.sample_length = (video_length - 1) * clip_step + 1
        
        # Setup sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.sample_length,
            pad_before=0, 
            pad_after=0,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.output_length = video_length  # Store desired output length
        
        # Setup transforms
        if self.spatial_transform == "center_crop_resize":
            print('Spatial transform: center crop and then resize')
            self.video_transform = transforms.Compose([
                transforms_video.CenterCropVideo(resolution),
                transforms.Resize(resolution)
            ])
        elif self.spatial_transform == "resize":
            print('Spatial transform: resize with no crop')
            self.video_transform = transforms.Resize((resolution, resolution))
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                transforms_video.RandomCropVideo(resolution),
            ])
        elif self.spatial_transform == "":
            self.video_transform = None
        else:
            raise NotImplementedError(f"Unknown spatial transform: {self.spatial_transform}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.sample_length,
            pad_before=0, 
            pad_after=0,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Calculate indices to maintain output video length
        indices = np.arange(0, self.sample_length, self.clip_step)
        
        # Process images
        image = np.moveaxis(sample['img'][indices], -1, 1)  # T, 3, H, W
        image = torch.from_numpy(image).float() / 255.0  # Convert to tensor and normalize to [0, 1]
        
        # Apply spatial transform if specified
        if self.video_transform is not None:
            image = self.video_transform(image)
            
        # Normalize to [-1, 1]
        image = 2.0 * image - 1.0
            
        # Process other data
        agent_pos = sample['state'][indices, :2].astype(np.float32)
        action = sample['action'][indices].astype(np.float32)

        data = {
            'obs': {
                'image': image,  # T, 3, H, W
                'agent_pos': agent_pos,  # T, 2
            },
            'action': action  # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, lambda x: x if torch.is_tensor(x) else torch.from_numpy(x))

        example = dict()
        example["image"] = torch_data['obs']['image'].permute(1, 0, 2, 3)
        example["frame_stride"] = self.clip_step
        example["action"] = torch_data["action"]

        # Add the first frame if the flag is set
        if self.include_first_frame:
            first_frame = torch_data['obs']['image'][0]  # Extract the first frame (3, H, W)
            example["first_frame"] = first_frame

        return example


def test():
    dataset = PushTVideoDataset(
        data_root='/home/ubuntu/Robot/LVDM/datasets/push_t',
        resolution=96,
        video_length=16,
        subset_split='test',
        spatial_transform='center_crop_resize',
        clip_step=2  # Sample every other frame
    )
    
    # Test basic functionality
    sample = dataset[0]
    print("Sample shapes:")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Frame Stride: {sample['frame_stride']}")
    # print(f"Image shape: {sample['obs']['image'].shape}")
    # print(f"Agent position shape: {sample['obs']['agent_pos'].shape}")
    # print(f"Action shape: {sample['action'].shape}")
    
    # Test different spatial transforms
    transforms_to_test = ['center_crop_resize', 'resize', 'random_crop', '']
    for transform in transforms_to_test:
        try:
            print(f"\nTesting {transform} transform:")
            test_dataset = PushTVideoDataset(
                data_root='/home/ubuntu/Robot/LVDM/datasets/push_t',
                resolution=96,
                video_length=16,
                subset_split='test',
                spatial_transform=transform,
                clip_step=1
            )
            test_sample = test_dataset[0]
            print(f"Image shape with {transform}: {test_sample['image'].shape}")
        except Exception as e:
            print(f"Error testing {transform}: {str(e)}")
    
    # Test validation split
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset length: {len(val_dataset)}")
    
    # Test different clip steps
    for clip_step in [1, 2, 4]:
        test_dataset = PushTVideoDataset(
            data_root='/home/ubuntu/Robot/LVDM/datasets/push_t',
            resolution=96,
            video_length=16,
            subset_split='test',
            spatial_transform='center_crop_resize',
            clip_step=clip_step
        )
        test_sample = test_dataset[0]
        print(f"\nClip step {clip_step}:")
        print(f"Image sequence length: {test_sample['image'].shape[0]}")


if __name__ == "__main__":
    test()