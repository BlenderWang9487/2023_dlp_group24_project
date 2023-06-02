from .my_diffusers import *
import pathlib as pl

class DoubleDenoisingRatioScheduler(nn.Module):
    def __init__(self, ratio_type = 'linear', time_steps = 1000, start_r = 0., end_r = 1.) -> None:
        super().__init__()

        # ratio is for the 'expert_unet_1' at timestep t
        if ratio_type == 'linear':
            self.register_buffer('ratio', torch.linspace(start_r, end_r, steps=time_steps))
        else:
            raise NotImplementedError()
        
    def get_ratio(self, t: torch.Tensor, batch=False):
        if batch:
            if t.dim() == 1:
                return self.ratio[t][:, None, None, None]
            elif t.dim() == 2:
                return self.ratio[t][:, :, None, None]
            else:
                raise RuntimeError()
        return self.ratio[t]


class MyDoubleDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    def __init__(self, unet1, unet2, scheduler, device='cuda'):
        super().__init__()
        self.my_device = device
        self.register_modules(expert_unet_1=unet1, expert_unet_2=unet2, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        condition: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        cfg_scale: Optional[float] = None,
        num_inference_steps: int = 1000,
        ratio_scheduler = DoubleDenoisingRatioScheduler(),
        return_dict: bool = True,
        **kwargs,
    ) -> Union[MyImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.expert_unet_1.sample_size, int):
            image_shape = (batch_size, self.expert_unet_1.in_channels, self.expert_unet_1.sample_size, self.expert_unet_1.sample_size)
        else:
            image_shape = (batch_size, self.expert_unet_1.in_channels, *self.expert_unet_1.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = torch.randn(image_shape, generator=generator)
            image = image.to(self.my_device)
        else:
            image = torch.randn(image_shape, generator=generator, device=self.my_device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        double_unet = DoubleUnet(expert_unet_1=self.expert_unet_1, expert_unet_2=self.expert_unet_2)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            t = t.to(self.my_device)
            ratios = ratio_scheduler.get_ratio(t=t)
            model_output = double_unet(image, t, ratio=ratios, class_labels=condition.float()).sample
            if cfg_scale is not None: # classifier free guidance
                uncond_output = double_unet(image, t, ratio=ratios, class_labels=None).sample
                model_output = torch.lerp(uncond_output, model_output, cfg_scale)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1) not unnormalize yet, we need to evaluate
        # image = image.cpu()

        if not return_dict:
            return (image,)

        return MyImagePipelineOutput(images=image)


class DoubleUnet(nn.Module):
    def __init__(self, expert_unet_1: MyUNet2DModel, expert_unet_2: MyUNet2DModel) -> None:
        super(DoubleUnet, self).__init__()

        self.expert_unet_1 = expert_unet_1
        self.expert_unet_2 = expert_unet_2

    @staticmethod
    def from_pretrained(pretrained_dir: str):
        p = pl.Path(pretrained_dir)
        unet_1 = MyUNet2DModel.from_pretrained(p / "export_unet_1")
        unet_2 = MyUNet2DModel.from_pretrained(p / "export_unet_2")

        return DoubleUnet(unet_1, unet_2)
    
    @staticmethod
    def from_unet_pretrained(pretrained_dir: str):
        unet_1 = MyUNet2DModel.from_pretrained(pretrained_dir)
        unet_2 = MyUNet2DModel.from_pretrained(pretrained_dir)

        return DoubleUnet(unet_1, unet_2)
    

    def save_pretrained(self, pretrained_dir: str):
        p = pl.Path(pretrained_dir)

        self.expert_unet_1.save_pretrained(p / "export_unet_1")
        self.expert_unet_2.save_pretrained(p / "export_unet_2")

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        ratio: Union[torch.Tensor, float],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        
        pred_1 = self.expert_unet_1(
            sample = sample, timestep = timestep, class_labels = class_labels, return_dict=False)
        pred_2 = self.expert_unet_2(
            sample = sample, timestep = timestep, class_labels = class_labels, return_dict=False)
        
        pred = pred_1[0] * ratio + pred_2[0] * (1. - ratio)

        if not return_dict:
            return (pred,)

        return UNet2DOutput(sample=pred)
        
