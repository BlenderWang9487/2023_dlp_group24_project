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
    image_size = 32
    uncond_prob = 0.1
    train_batch_size = 64
    # train_batch_size = 16
    eval_batch_size = 10
    num_epochs = 200
    # num_epochs = 70
    # gradient_accumulation_steps = 4
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    mixed_precision = "fp16"
    # output_dir = "ckpt/cifar10/0530"
    output_dir = "ckpt/cifar10/0531"
    num_workers = 6
    device = 'cuda'
    scheduler_type = 'squaredcos_cap_v2'

    push_to_hub = False
    overwrite_output_dir = True
    seed = 9487

class evaluation_model:
    pass


@torch.no_grad()
def evaluate(config: TrainingConfig, epoch, pipeline: MyDDPMPipeline, cfg_scale = None):
    logger = logging.getLogger('Cond_DDPM')

    def unnormalize_to_zero_to_one(t):
        return torch.clamp((t + 1) * 0.5, min=0., max=1.)

    cond = torch.from_numpy(np.eye(10)).to(config.device)
    images = pipeline(
        batch_size=config.eval_batch_size,
        condition=cond,
        generator=torch.cuda.manual_seed(config.seed),
        cfg_scale=cfg_scale
    ).images

    # Make a grid out of the images
    images = unnormalize_to_zero_to_one(images)

    image_grid = make_grid(images, nrows=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")

    os.makedirs(test_dir, exist_ok=True)

    cfg_prefix = '' if cfg_scale is None else f'cfg{cfg_scale}_'
    save_image(image_grid, f"{test_dir}/{cfg_prefix}{epoch:04d}.png")

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
            # transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def to_onehot(y, class_num=10):
        y_vec = torch.zeros((class_num, ))
        y_vec[y] = 1.
        return y_vec

    data = datasets.CIFAR10(root='dataset/cifar10', transform=preprocess, target_transform=to_onehot, download=True)

    train_dataloader = DataLoader(
        data, batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True)

    # model = MyUNet2DConditionModel(
    #     sample_size=config.image_size,  # the target image resolution
    #     in_channels=3,  # the number of input channels, 3 for RGB images
    #     out_channels=3,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(128, 128, 256, 256, 512),  # the number of output channels for each UNet block
    #     down_block_types=(
    #         "DownBlock2D",  # a regular ResNet downsampling block
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    #         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    #         # "DownBlock2D",
    #     ),
    #     up_block_types=(
    #         # "UpBlock2D",  # a regular ResNet upsampling block
    #         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
    #         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
    #         "UpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D",
    #     ),
    #     num_class_embeds=10,
    #     cross_attention_dim=512
    # )
    model = MyUNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        num_class_embeds=10,
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
            total_loss /= len(train_dataloader)
            # logger.info(f"{logs}, step={global_step}")
            logger.info(f"epoch {epoch}, step={global_step}, loss avg={total_loss}")

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = MyDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir)
                    logger.info(f"Save epoch#{epoch} model to {str(config.output_dir)}")

    train_loop(
        config=config, 
        model=model, 
        noise_scheduler=noise_scheduler, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        lr_scheduler=lr_scheduler, writer=writer)

if __name__ == "__main__":
    Train()