from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers import DDPMScheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusers.optimization import get_cosine_schedule_with_warmup
from model.double_diffusion import DoubleUnet, MyDoubleDDPMPipeline, DoubleDenoisingRatioScheduler
import torch
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    image_size = 32
    uncond_prob = 0.0
    train_batch_size = 128
    eval_batch_size = 100
    num_epochs = 200
    # num_epochs = 30
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    # lr_warmup_steps = 391
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    mixed_precision = "fp16"
    # output_dir = "ckpt/cifar10/0602/double"
    # output_dir = "ckpt/cifar10/0603/double"
    output_dir = "ckpt/cifar10/0605/double"
    # unet_pretrained = "ckpt/cifar10/0601/unet"
    unet_pretrained = "ckpt/cifar10/init_model"
    unet_pretrained2 = "ckpt/cifar10/init_model_2"
    num_workers = 4
    device = 'cuda'
    scheduler_type = 'squaredcos_cap_v2'
    # ratio_scheduler_type = 'linear'
    # ratio_scheduler_type = 'sigmoid'
    ratio_scheduler_type = 'learned'

    push_to_hub = False
    overwrite_output_dir = True
    seed = 9487

class evaluation_model:
    pass


@torch.no_grad()
def evaluate(
        config: TrainingConfig,
        epoch,
        pipeline: MyDoubleDDPMPipeline,
        ratio_scheduler: DoubleDenoisingRatioScheduler,
        cfg_scale = None,
    ):
    logger = logging.getLogger('Cond_DDPM')

    def unnormalize_to_zero_to_one(t):
        return torch.clamp((t + 1) * 0.5, min=0., max=1.)

    class_labels = np.zeros((100, 10))
    class_labels[np.arange(100), np.repeat(np.arange(10), 10)] = 1. # 100 images, 10 for each classes
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
        fig.savefig(ratio_dir / f"{epoch:04d}.png")
        
    test_dir.mkdir(parents=True, exist_ok=True)

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

    model = DoubleUnet.from_unet_pretrained(config.unet_pretrained, config.unet_pretrained2)
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
            total_loss /= len(train_dataloader)
            # logger.info(f"{logs}, step={global_step}")
            logger.info(f"epoch {epoch}, step={global_step}, loss avg={total_loss}")

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                if (epoch) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    unwap_m: DoubleUnet = accelerator.unwrap_model(model)
                    pipeline = MyDoubleDDPMPipeline(
                        unet1=unwap_m.expert_unet_1,
                        unet2=unwap_m.expert_unet_2,
                        scheduler=noise_scheduler)
                    evaluate(config, epoch, pipeline, ratio_scheduler)

                if (epoch) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    unwap_m: DoubleUnet = accelerator.unwrap_model(model)
                    unwap_m.save_pretrained(config.output_dir)
                    noise_scheduler.save_pretrained(config.output_dir)
                    torch.save(accelerator.unwrap_model(ratio_scheduler).state_dict(), Path(config.output_dir) / "ratio_scheduler.pt")
                    logger.info(f"Save epoch#{epoch} model to {str(config.output_dir)}")

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