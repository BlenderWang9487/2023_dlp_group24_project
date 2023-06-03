from model.my_diffusers import MyDDPMPipeline, MyUNet2DModel
from model.double_diffusion import DoubleUnet, DoubleDenoisingRatioScheduler, MyDoubleDDPMPipeline
from diffusers import DDPMScheduler
import torch
import numpy as np
import pathlib as pl

def separated_sample():
    from train_simple import TrainingConfig, evaluate
    config = TrainingConfig()
    config.seed = 1212

    for expert in [1, 2]:
        config.output_dir = f'ckpt/cifar10/0602/test/expert_{expert}'

        model = MyUNet2DModel.from_pretrained(f"ckpt/cifar10/0602/double/export_unet_{expert}").to(config.device)
        scheduler = DDPMScheduler(beta_schedule=config.scheduler_type)

        pipeline = MyDDPMPipeline(unet=model, scheduler=scheduler, device=config.device)

        evaluate(config, config.seed, pipeline)

def different_condition():
    from torchvision.utils import make_grid, save_image
    from train_double import TrainingConfig
    @torch.no_grad()
    def evaluate_two_cond(
            config: TrainingConfig,
            epoch,
            pipeline: MyDoubleDDPMPipeline,
            ratio_scheduler: DoubleDenoisingRatioScheduler,
            class_1: int = 0,
            class_2: int = 1,
            cfg_scale = None,
        ):

        def unnormalize_to_zero_to_one(t):
            return torch.clamp((t + 1) * 0.5, min=0., max=1.)
        
        assert config.eval_batch_size % 2 == 0, "eval batch size should be even number!"

        class_labels = np.zeros((config.eval_batch_size, 10))
        half = config.eval_batch_size // 2
        class_labels[:half, class_1] = 1. # first half images use class 1 as condition 1
        class_labels[half:, class_2] = 1. # last half images use class 2 as condition 1

        cond1 = torch.from_numpy(class_labels).to(config.device)
        cond2 = torch.from_numpy(class_labels[::-1, :].copy()).to(config.device)

        images = pipeline(
            batch_size=config.eval_batch_size,
            condition=cond1,
            condition_alternative=cond2,
            generator=torch.cuda.manual_seed(config.seed),
            cfg_scale=cfg_scale,
            ratio_scheduler=ratio_scheduler,
        ).images

        # Make a grid out of the images
        images = unnormalize_to_zero_to_one(images)

        image_grid = make_grid(images, nrow=half)

        # Save the images
        test_dir = pl.Path(config.output_dir) / "samples"
        test_dir.mkdir(parents=True, exist_ok=True)

        cfg_prefix = '' if cfg_scale is None else f'cfg{cfg_scale}_'
        save_image(image_grid, f"{str(test_dir)}/{cfg_prefix}{epoch:04d}.png")

    config = TrainingConfig()
    config.seed = 1214
    config.output_dir = "ckpt/cifar10/0602/test/two_cond"
    config.eval_batch_size = 20

    model = DoubleUnet.from_pretrained("ckpt/cifar10/0602/double").to(config.device)
    scheduler = DDPMScheduler(beta_schedule=config.scheduler_type)
    pipeline = MyDoubleDDPMPipeline(unet1=model.expert_unet_1, unet2=model.expert_unet_2, scheduler=scheduler)
    ratio_scheduler = DoubleDenoisingRatioScheduler()


    evaluate_two_cond(config, config.seed, pipeline, ratio_scheduler,
        # class_1=7, # horse
        class_1=2,
        # class_2=9, # truck
        class_2=9,
        )
    
if __name__ == "__main__":
    different_condition()

