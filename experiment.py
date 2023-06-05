from model.my_diffusers import MyDDPMPipeline, MyUNet2DModel
from model.double_diffusion import DoubleUnet, DoubleDenoisingRatioScheduler, MyDoubleDDPMPipeline
from diffusers import DDPMScheduler
import torch
import numpy as np
import pathlib as pl
import argparse

def separated_sample(args):
    from train_simple import TrainingConfig, evaluate
    config = TrainingConfig()
    config.seed = args.seed

    for expert in [1, 2]:
        config.output_dir = args.output_dir / f'expert_{expert}'

        model = MyUNet2DModel.from_pretrained(args.pretrained / f"export_unet_{expert}").to(config.device)
        scheduler = DDPMScheduler(beta_schedule=config.scheduler_type)

        pipeline = MyDDPMPipeline(unet=model, scheduler=scheduler, device=config.device)

        evaluate(config, config.seed, pipeline)

def different_condition(args):
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
        test_dir = config.output_dir / "samples"
        test_dir.mkdir(parents=True, exist_ok=True)

        cfg_prefix = '' if cfg_scale is None else f'cfg{cfg_scale}_'
        save_image(image_grid, f"{str(test_dir)}/{cfg_prefix}{epoch:04d}.png")

    config = TrainingConfig()
    config.seed = args.seed
    config.output_dir = args.output_dir / "two_cond"
    config.eval_batch_size = 20

    model = DoubleUnet.from_pretrained(args.pretrained).to(config.device)
    scheduler = DDPMScheduler(beta_schedule=config.scheduler_type)
    pipeline = MyDoubleDDPMPipeline(unet1=model.expert_unet_1, unet2=model.expert_unet_2, scheduler=scheduler)
    ratio_scheduler = DoubleDenoisingRatioScheduler(ratio_type=config.ratio_scheduler_type)


    evaluate_two_cond(config, config.seed, pipeline, ratio_scheduler,
        # class_1=7, # horse
        class_1=args.class1,
        # class_2=9, # truck
        class_2=args.class2,
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment')
    parser.add_argument('-e', '--experiment', help='which experiment?', choices=["different_condition", "separated_sample"], default='separated_sample')
    parser.add_argument('-p', '--pretrained', help='pretrained double diffusion dir', type=pl.Path, default='ckpt/cifar10/0603/double')
    parser.add_argument('-o', '--output_dir', help='output img dir', type=pl.Path, default='/tmp')
    parser.add_argument('-1', '--class1', help='class 1 for two cond', type=int, default=0)
    parser.add_argument('-2', '--class2', help='class 2 for two cond', type=int, default=1)
    parser.add_argument('--seed', help='random seed for generation', type=int, default=1212)

    args = parser.parse_args()

    exprs = [different_condition, separated_sample]

    for expr in exprs:
        if args.experiment == expr.__name__:
            expr(args)
            exit(0)
    
    raise NotImplementedError()
    


