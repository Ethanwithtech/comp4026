import torch
import torch.nn as nn
from diffusers import UNet2DModel  

class ConditionalUNet(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=9,       
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, noisy_image, timestep, landmark_img, masked_img):
        x = torch.cat([noisy_image, landmark_img, masked_img], dim=1) 
        
        noise_pred = self.unet(x, timestep).sample
        
        return noise_pred