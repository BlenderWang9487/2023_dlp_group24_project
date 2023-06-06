'''
Some helper functions
'''
from torch.nn import Module
from typing import Optional

def get_num_parameters(model: Module, only_trainable = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_baseline_unet(pretrained: Optional[str]=None, image_size=32, num_classes=10, half=False):
    from model.double_diffusion import MyUNet2DModel
    if pretrained is not None:
        return MyUNet2DModel.from_pretrained(pretrained)
    
    return MyUNet2DModel(
        sample_size=image_size,
        block_out_channels=(128, 128, 256, 256, 512, 512) if not half else (64, 64, 128, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        in_channels=3,
        out_channels=3,
        num_class_embeds=num_classes,
    )