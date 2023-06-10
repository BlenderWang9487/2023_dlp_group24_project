import os
import csv
import random
import numpy as np
import argparse
from tqdm.auto import tqdm
import pathlib as pl
from pathlib import Path
from typing import Union, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusers import DDPMScheduler, DDIMScheduler
from model.my_diffusers import MyDDPMPipeline, MyUNet2DModel, MyConditionedUNet
from model.double_diffusion import MyDoubleDDPMPipeline, DoubleDenoisingRatioScheduler, DoubleUnet

def unnormalize_to_zero_to_one(t):
    return torch.clamp((t + 1) * 0.5, min=0., max=1.)

def get_args():
    parser = argparse.ArgumentParser("FID evaluator")
    parser.add_argument('-p', '--pretrained', type=Path,  required=True, help="the pretrained model dir")
    parser.add_argument('--single', action='store_true', help='is the model single unet or double unet')
    parser.add_argument('--dataset', choices=['lsun', 'cifar'], default='lsun', help='dataset that the model was trained on')
    parser.add_argument('--data_dir', type=Path, default='dataset/lsun', help='dataset root directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for fid update')
    parser.add_argument('--sample_count', type=int, default=10000, help='number of images to sample by model')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model on')
    parser.add_argument('--noise_type', type=str, default='squaredcos_cap_v2', help='noise scheduler')
    parser.add_argument('--ratio_type', type=str, choices=['linear', 'sigmoid', 'learned'], default='linear', help='ratio scheduler type')
    parser.add_argument('--seed', type=int, default=9487, help='seed for reproduction')
    parser.add_argument('--img_size', type=int, default=64, help='size of the images that model was trained on')
    parser.add_argument('--use_ddim', action='store_true', help='use DDIM as scheduler to accelerate sampling speed')
    parser.add_argument('--eta', type=float, default=0.0, help='The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM')
    parser.add_argument('-o', '--output_img', type=Path, default=None, help='Dir to save imgs sampled, default is ***NOT TO SAVE ANY***')
    parser.add_argument('--big_sister', action='store_true', help='Use big sister model architecture')
    return parser.parse_args()

def calculate_fid(
        eval_loader: DataLoader,
        pipeline: Union[MyDDPMPipeline, MyDoubleDDPMPipeline],
        batch_size=128,
        sample_count=1000,
        class_num=10,
        device='cpu',
        seed=None,
        eta=0., # if not using DDIM, this will be ignored
        img_dir: Path=None,
        ratio_scheduler: DoubleDenoisingRatioScheduler=None
    ):
    fid_model = FrechetInceptionDistance(normalize=True).to(device) # image value is between [0, 1]
    
    real_labels = []
    # update real image from eval_loader
    for img, label in tqdm(eval_loader, desc="Update eval dataset features: "):
        img = img.to(device)
        fid_model.update(img, real=True)
        real_labels.append(label.view(-1).numpy())

    real_labels = np.resize(np.concatenate(real_labels), sample_count)

    one_hot_matrix = np.eye(class_num)

    step = 0
    while step * batch_size < sample_count:
        cond = torch.from_numpy(
            one_hot_matrix[real_labels[step * batch_size:(step + 1) * batch_size]]
        ).to(device).float()

        gen = None
        if seed is not None:
            gen = torch.manual_seed(seed + step) if device == 'cpu' else torch.cuda.manual_seed(seed + step)
        
        fake_img = pipeline(
            batch_size=cond.shape[0],
            condition=cond,
            generator=gen,
            eta=eta,
            ratio_scheduler=ratio_scheduler
        ).images
        fake_img = unnormalize_to_zero_to_one(fake_img)
        fid_model.update(fake_img, real=False)
        step += 1

        updated_images_num = min(step * batch_size, sample_count)

        addition_msg = ""
        if img_dir is not None:
            grid = make_grid(fake_img, nrow=16)
            img_file = img_dir / f"{step:04d}.png"
            save_image(grid, img_file)
            addition_msg = f"Saving fake images to {img_file}."
        print(f"Updated {updated_images_num}/{sample_count} ({updated_images_num/sample_count:%}) fake images. {addition_msg}\n")

    return float(fid_model.compute())


if __name__ == "__main__":
    args = get_args()
    print(f"Args: {args}")
    noise_scheduler = DDPMScheduler(beta_schedule=args.noise_type) if not args.use_ddim else DDIMScheduler(beta_schedule=args.noise_type)
    if args.single:
        pretrain_path = args.pretrained
        if args.big_sister:
            model = MyConditionedUNet.from_pretrained(pretrain_path)
        else:
            model = MyUNet2DModel.from_pretrained(pretrain_path)
        model = model.to(args.device)
        pipeline = MyDDPMPipeline(unet=model, scheduler=noise_scheduler, device=args.device)
    else:
        model = DoubleUnet.from_pretrained(args.pretrained).to(args.device)
        ratio_scheduler = DoubleDenoisingRatioScheduler(ratio_type=args.ratio_type)
        if args.ratio_type == 'learned':
            ratio_scheduler.load_state_dict(torch.load(args.pretrained / 'ratio_scheduler.pt'))
        ratio_scheduler = ratio_scheduler.to(args.device)
        pipeline = MyDoubleDDPMPipeline(unet1=model.expert_unet_1, unet2=model.expert_unet_2, scheduler=noise_scheduler, device=args.device)

    
    preprocess = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    if args.dataset == 'lsun':
        classes = ['church_outdoor_val','classroom_val','conference_room_val','dining_room_val']
        dataset = datasets.LSUN(root=args.data_dir, transform=preprocess, classes=classes)
    elif args.dataset == 'cifar':
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=preprocess, download=True)
    print(f"Eval dataset size: {len(dataset)}")
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    if args.output_img is not None:
        args.output_img.mkdir(parents=True, exist_ok=True)
    fid = calculate_fid(
        eval_loader=eval_loader,
        pipeline=pipeline,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        class_num=4 if args.dataset == 'lsun' else 10,
        device=args.device,
        seed=args.seed,
        eta=args.eta,
        img_dir=args.output_img,
        ratio_scheduler=ratio_scheduler if not args.single else None
        )
    print(f"FID: {fid}")
    if args.output_img is not None:
        with open(args.output_img / "parametes_and_fid.txt", 'w') as f:
            f.write(f"Args: {args}\n\n")
            f.write(f"FID: {fid}\n")



