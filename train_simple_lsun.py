from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers import DDPMScheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusers.optimization import get_cosine_schedule_with_warmup
from model.my_diffusers import MyDDPMPipeline, MyUNet2DConditionModel, MyUNet2DModel
import torch
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os
import random
import logging
import numpy as np

@dataclass
class TrainingConfig:
    image_size = 64
    uncond_prob = 0.0
    train_batch_size = 64
    eval_batch_size = 100
    class_num = 4
    num_epochs = 30
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    save_image_iter = 5000
    save_model_iter = 10000
    log_iter = 1000
    mixed_precision = "fp16"
    output_dir = "ckpt/cifar10/0608/lsun_baseline"
    num_workers = 4
    device = 'cuda'
    scheduler_type = 'squaredcos_cap_v2'

    push_to_hub = False
    overwrite_output_dir = True
    seed = 9487

class evaluation_model:
    pass


@torch.no_grad()
def evaluate(config: TrainingConfig, iteration, pipeline: MyDDPMPipeline, cfg_scale = None):
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
        cfg_scale=cfg_scale
    ).images

    # Make a grid out of the images
    images = unnormalize_to_zero_to_one(images)

    image_grid = make_grid(images, nrow=config.eval_batch_size // config.class_num)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")

    os.makedirs(test_dir, exist_ok=True)

    cfg_prefix = '' if cfg_scale is None else f'cfg{cfg_scale}_'
    save_image(image_grid, f"{test_dir}/{cfg_prefix}{iteration:06d}.png")

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
        # root='dataset/lsun', 
        root='/mnt/ssd/blender/lsun', 
        transform=preprocess, 
        target_transform=to_onehot, 
        classes=['church_outdoor_train','classroom_train','conference_room_train','dining_room_train'])

    train_dataloader = DataLoader(
        data, batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True)

    model = MyUNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
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
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        num_class_embeds=config.class_num,
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=config.scheduler_type)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    criterion = torch.nn.MSELoss()

    def train_loop(
            config: TrainingConfig,
            model: Union[MyUNet2DConditionModel, MyUNet2DModel],
            noise_scheduler: DDPMScheduler,
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
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0
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

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, class_labels=cond_labels, return_dict=False)[0]
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
                        pipeline = MyDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                        evaluate(config, global_step, pipeline)

                    if (global_step) % config.save_model_iter == 0 or epoch == config.num_epochs - 1:
                        pipeline = MyDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                        save_dir = Path(config.output_dir) / "weight" / f"{global_step:06d}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        pipeline.save_pretrained(save_dir)
                        logger.info(f"Save iter#{global_step} model to {str(save_dir)}")
            
                    if global_step % config.log_iter == 0:
                        total_loss /= config.log_iter
                        logger.info(f"step={global_step}, loss avg={total_loss}")
                        total_loss = 0.

    train_loop(
        config=config, 
        model=model, 
        noise_scheduler=noise_scheduler, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        lr_scheduler=lr_scheduler, writer=writer)

if __name__ == "__main__":
    Train()