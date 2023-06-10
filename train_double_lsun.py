from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers import DDPMScheduler, DDIMScheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusers.optimization import get_cosine_schedule_with_warmup
from model.my_diffusers import MyDDPMPipeline, MyUNet2DConditionModel, MyUNet2DModel, MyConditionedUNet
from model.double_diffusion import DoubleUnet, DoubleDenoisingRatioScheduler, MyDoubleDDPMPipeline
import torch
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import os
import random
import logging
import numpy as np

@dataclass
class TrainingConfig:
    image_size = 64
    uncond_prob = 0.0
    train_batch_size = 32
    eval_batch_size = 100
    class_num = 4
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    save_image_iter = 5000
    save_model_iter = 5000
    log_iter = 1000
    unet_pretrained = "ckpt/lsun/base_735000/unet"
    mixed_precision = "fp16"
    output_dir = "ckpt/lsun/0610/double"
    num_workers = 4
    device = 'cuda'
    scheduler_type = 'squaredcos_cap_v2'
    # ratio_scheduler_type = 'linear'
    # ratio_scheduler_type = 'sigmoid'
    ratio_scheduler_type = 'learned'
    start_step = 735000

    push_to_hub = False
    overwrite_output_dir = True
    seed = 9487

class evaluation_model:
    pass


@torch.no_grad()
def evaluate(
        config: TrainingConfig,
        iteration,
        pipeline: MyDoubleDDPMPipeline,
        ratio_scheduler: DoubleDenoisingRatioScheduler,
        cfg_scale = None,
    ):
    logger = logging.getLogger('Cond_DDPM')

    def unnormalize_to_zero_to_one(t):
        return torch.clamp((t + 1) * 0.5, min=0., max=1.)

    class_labels = np.zeros((config.eval_batch_size, config.class_num))
    class_labels[np.arange(config.eval_batch_size), np.repeat(np.arange(config.class_num), config.eval_batch_size // config.class_num)] = 1. # 100 images, 25 for each classes
    cond = torch.from_numpy(class_labels).to(config.device)
    images = pipeline(
        batch_size=config.eval_batch_size,
        condition=cond,
        generator=torch.cuda.manual_seed(config.seed),
        cfg_scale=cfg_scale,
        ratio_scheduler=ratio_scheduler,
    ).images

    # Make a grid out of the images
    images = unnormalize_to_zero_to_one(images)

    image_grid = make_grid(images, nrow=10)

    # Save the images
    test_dir = Path(config.output_dir) / "samples"

    # check ratio
    if ratio_scheduler.ratio_type == 'learned':
        ratio_dir = Path(config.output_dir) / "ratio"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 2)
        t = torch.arange(1000)
        ratio = ratio_scheduler.get_ratio(t.to(config.device), True).squeeze().cpu().numpy()
        t = t.numpy()
        ax.set_ylim(0, 1)
        ax.fill_between(t, 0, ratio, label='expert1')
        ax.fill_between(t, ratio, 1, label='expert2')
        ax.legend()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Ratio (of expert 1)')
        fig.tight_layout()
        fig.savefig(ratio_dir / f"{iteration:04d}.png")
        
    test_dir.mkdir(parents=True, exist_ok=True)

    cfg_prefix = '' if cfg_scale is None else f'cfg{cfg_scale}_'
    save_image(image_grid, f"{test_dir}/{cfg_prefix}{iteration:04d}.png")

def Train():
    config = TrainingConfig()
    logdir = Path(config.output_dir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(logdir / "log.txt"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logging.info("Start training DDPM by diffusers")
    logger = logging.getLogger('Cond_DDPM')
    writer = SummaryWriter(Path(config.output_dir) / "tb")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def to_onehot(y, class_num=config.class_num):
        y_vec = torch.zeros((class_num, ))
        y_vec[y] = 1.
        return y_vec

    data = datasets.LSUN(
        root='/mnt/ssd/blender/lsun', 
        transform=preprocess, 
        target_transform=to_onehot, 
        classes=['church_outdoor_train','classroom_train','conference_room_train','dining_room_train'])

    train_dataloader = DataLoader(
        data, batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True)

    model = DoubleUnet.from_unet_pretrained(config.unet_pretrained, ModelType=MyConditionedUNet)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=config.scheduler_type)
    ratio_scheduler = DoubleDenoisingRatioScheduler(ratio_type=config.ratio_scheduler_type, time_steps=1000)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(ratio_scheduler.parameters()), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    criterion = torch.nn.MSELoss()

    def train_loop(
            config: TrainingConfig,
            model: DoubleUnet,
            noise_scheduler: DDPMScheduler,
            ratio_scheduler: DoubleDenoisingRatioScheduler,
            optimizer: torch.optim.AdamW,
            train_dataloader: DataLoader,
            lr_scheduler, writer: SummaryWriter):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler, ratio_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, ratio_scheduler
        )

        global_step = config.start_step
        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            total_loss = 0.
            for step, (X, y) in enumerate(train_dataloader):
                clean_images = X
                cond_labels = None if random.random() < config.uncond_prob else y
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long()
                ratios = ratio_scheduler.get_ratio(t=timesteps, batch=True)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, ratio=ratios, class_labels=cond_labels, return_dict=False)[0]
                    loss = criterion(noise_pred, noise)
                    accelerator.backward(loss)
                    # if accelerator.sync_gradients:
                    #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                writer.add_scalar("Train loss", loss.detach().item(), global_step=global_step)

                total_loss += loss.detach().item()
                progress_bar.set_postfix(**logs)
                global_step += 1

                # After each epoch you optionally sample some demo images with evaluate() and save the model
                if accelerator.is_main_process:
                    if (global_step) % config.save_image_iter == 0 or epoch == config.num_epochs - 1:
                        unwap_m: DoubleUnet = accelerator.unwrap_model(model)
                        pipeline = MyDoubleDDPMPipeline(
                            unet1=unwap_m.expert_unet_1,
                            unet2=unwap_m.expert_unet_2,
                            scheduler=DDIMScheduler(num_train_timesteps=1000, beta_schedule=config.scheduler_type))
                        evaluate(config, global_step, pipeline, ratio_scheduler)

                    if (global_step) % config.save_model_iter == 0 or epoch == config.num_epochs - 1:
                        unwap_m: DoubleUnet = accelerator.unwrap_model(model)
                        save_dir = Path(config.output_dir) / "weight" / f"{global_step:06d}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        unwap_m.save_pretrained(save_dir)
                        noise_scheduler.save_pretrained(save_dir)
                        torch.save(accelerator.unwrap_model(ratio_scheduler).state_dict(), save_dir / "ratio_scheduler.pt")
                        logger.info(f"Save iter#{global_step} model to {str(save_dir)}")
            
                    if global_step % config.log_iter == 0:
                        total_loss /= config.log_iter
                        logger.info(f"step={global_step}, loss avg={total_loss}")
                        total_loss = 0.

    train_loop(
        config=config, 
        model=model, 
        noise_scheduler=noise_scheduler, 
        ratio_scheduler=ratio_scheduler,
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        lr_scheduler=lr_scheduler, writer=writer)

if __name__ == "__main__":
    Train()