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

from diffusers import DDPMScheduler
from model.my_diffusers import MyDDPMPipeline, MyUNet2DModel
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
    parser.add_argument('--ratio_type', type=str, default='linear', help='ratio scheduler type')
    parser.add_argument('--seed', type=int, default=9487, help='seed for reproduction')
    parser.add_argument('--img_size', type=int, default=64, help='size of the images that model was trained on')
    return parser.parse_args()

def calculate_fid(
        eval_loader: DataLoader,
        pipeline: Union[MyDDPMPipeline, MyDoubleDDPMPipeline],
        batch_size=128,
        sample_count=1000,
        class_num=10,
        device='cpu',
        seed=None
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
        ).images
        fid_model.update(unnormalize_to_zero_to_one(fake_img), real=False)
        step += 1

    return float(fid_model.compute())


if __name__ == "__main__":
    args = get_args()
    noise_scheduler = DDPMScheduler(beta_schedule=args.noise_type)
    if args.single:
        model = MyUNet2DModel.from_pretrained(args.pretrained).to(args.device)
        pipeline = MyDDPMPipeline(unet=model, scheduler=noise_scheduler, device=args.device)
    else:
        model = DoubleUnet.from_pretrained(args.pretrained)
        ratio_scheduler = DoubleDenoisingRatioScheduler(ratio_type=args.ratio_type)
        if args.ratio_type == 'learned':
            ratio_scheduler.load_state_dict(torch.load(args.pretrained / 'ratio_scheduler.pt'))
        pipeline = MyDoubleDDPMPipeline(unet1=model.expert_unet_1, unet2=model.expert_unet_2, device=args.device)

    
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
    fid = calculate_fid(
        eval_loader=eval_loader,
        pipeline=pipeline,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        class_num=4 if args.dataset == 'lsun' else 10,
        device=args.device,
        seed=args.seed
        )
    print(f"FID: {fid}")
    print(f"Args: {args}")



