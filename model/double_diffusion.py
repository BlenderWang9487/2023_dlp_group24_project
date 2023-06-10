from .my_diffusers import *
import pathlib as pl

class DoubleDenoisingRatioScheduler(nn.Module):
    def __init__(
            self,
            ratio_type = 'linear',
            time_steps = 1000,
            start_r = 0.,
            end_r = 1.,
            start_s = -15.,
            end_s = 5.,
            time_proj_size = 256,
            time_emb_size = 256,
        ) -> None:
        super().__init__()

        # ratio is for the 'expert_unet_1' at timestep t
        self.ratio_type = ratio_type
        if ratio_type == 'linear':
            self.register_buffer('ratio', torch.linspace(start_r, end_r, steps=time_steps))
        elif ratio_type == 'sigmoid':
            self.register_buffer('ratio', torch.sigmoid(torch.linspace(start_s, end_s, steps=time_steps)))
        elif ratio_type == 'learned':
            self.ratio = nn.Sequential(
                Timesteps(time_proj_size, True, 0),
                TimestepEmbedding(time_proj_size, time_emb_size, out_dim=1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError()
        
    def get_ratio(self, t: torch.Tensor, batch=False):
        if batch:
            if isinstance(self.ratio, nn.Module): # learned ratio
                return self.ratio(t.view(-1))[:, :, None, None]
            else:
                return self.ratio[t.view(-1)][:, None, None, None]
        return self.ratio[t] if not isinstance(self.ratio, nn.Module) else self.ratio(t.unsqueeze(0)).squeeze()


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
        condition_alternative: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.,
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
        use_ddim = isinstance(self.scheduler, DDIMScheduler)
        if use_ddim:
            num_inference_steps = 50

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
            model_output = double_unet(image, t, ratio=ratios, class_labels=condition.float(),
                                        class_labels_alternateive=condition_alternative.float() if condition_alternative is not None else None).sample
            if cfg_scale is not None: # classifier free guidance
                uncond_output = double_unet(image, t, ratio=ratios, class_labels=None).sample
                model_output = torch.lerp(uncond_output, model_output, cfg_scale)

            # 2. compute previous image: x_t -> x_t-1
            if use_ddim:
                image = self.scheduler.step(model_output, t, image, eta=eta, generator=generator).prev_sample
            else:
                image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1) not unnormalize yet, we need to evaluate
        # image = image.cpu()

        if not return_dict:
            return (image,)

        return MyImagePipelineOutput(images=image)

class YNetDDPMPipeline(DiffusionPipeline):
    def __init__(self, ynet, scheduler, device='cuda'):
        super().__init__()
        self.my_device = device
        self.register_modules(ynet=ynet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        condition: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.,
        num_inference_steps: int = 1000,
        return_dict: bool = True,
        sample_expert_idx: int = None,
        **kwargs,
    ) -> Union[MyImagePipelineOutput, Tuple]:
        use_ddim = isinstance(self.scheduler, DDIMScheduler)
        if use_ddim:
            num_inference_steps = 50

        # Sample gaussian noise to begin loop
        if isinstance(self.ynet.sample_size, int):
            image_shape = (batch_size, self.ynet.in_channels, self.ynet.sample_size, self.ynet.sample_size)
        else:
            image_shape = (batch_size, self.ynet.in_channels, *self.ynet.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = torch.randn(image_shape, generator=generator)
            image = image.to(self.my_device)
        else:
            image = torch.randn(image_shape, generator=generator, device=self.my_device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if sample_expert_idx is None: # normal sampling
                model_output = self.ynet(image, t.to(self.my_device), class_labels=condition.float()).sample
            else: # sample from certain expert
                assert sample_expert_idx in [0, 1]
                model_output = self.ynet(
                    image, t.to(self.my_device), class_labels=condition.float(),
                    separated_sample=True)[sample_expert_idx]

            # 2. compute previous image: x_t -> x_t-1
            if use_ddim:
                image = self.scheduler.step(model_output, t, image, eta=eta, generator=generator).prev_sample
            else:
                image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        if not return_dict:
            return (image,)

        return MyImagePipelineOutput(images=image)


class DoubleUnet(nn.Module):
    def __init__(self, expert_unet_1: MyUNet2DModel, expert_unet_2: MyUNet2DModel) -> None:
        super(DoubleUnet, self).__init__()

        self.expert_unet_1 = expert_unet_1
        self.expert_unet_2 = expert_unet_2

    @staticmethod
    def from_pretrained(pretrained_dir: str, ModelType = MyUNet2DModel):
        p = pl.Path(pretrained_dir)
        unet_1 = ModelType.from_pretrained(p / "export_unet_1")
        unet_2 = ModelType.from_pretrained(p / "export_unet_2")

        return DoubleUnet(unet_1, unet_2)
    
    @staticmethod
    def from_unet_pretrained(pretrained_dir: str, pretrained_dir2 = None, ModelType = MyUNet2DModel):
        if pretrained_dir2 is None:
            pretrained_dir2 = pretrained_dir
        unet_1 = ModelType.from_pretrained(pretrained_dir)
        unet_2 = ModelType.from_pretrained(pretrained_dir2)

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
        class_labels_alternateive: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        
        pred_1 = self.expert_unet_1(
            sample = sample, timestep = timestep, class_labels = class_labels, return_dict=False)
        pred_2 = self.expert_unet_2(
            sample = sample, timestep = timestep, class_labels = class_labels if class_labels_alternateive is None else class_labels_alternateive, return_dict=False)
        
        pred = pred_1[0] * ratio + pred_2[0] * (1. - ratio)

        if not return_dict:
            return (pred,)

        return UNet2DOutput(sample=pred)
        
class YNet2DModel(ModelMixin, ConfigMixin):
    r"""
    YNet2DModel is a 2D UNet model that takes in a noisy sample and a timestep and returns 2 sample shaped outputs.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`True`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        ratio_scheduler: The ratio scheduler.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        num_class_embeds = 10,
        ratio_scheduler_kwargs: dict = {},
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # ratio scheduler
        self.ratio_scheduler = DoubleDenoisingRatioScheduler(**ratio_scheduler_kwargs)

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        self.class_embedding = nn.Linear(num_class_embeds, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks_1 = nn.ModuleList([])
        self.up_blocks_2 = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block_1 = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            up_block_2 = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks_1.append(up_block_1)
            self.up_blocks_2.append(up_block_2)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out_1 = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_norm_out_2 = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out_1 = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        self.conv_out_2 = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        separated_sample: bool = False,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        ratio = self.ratio_scheduler.get_ratio(timesteps, batch=True)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        if class_labels is not None:
            emb += self.class_embedding(class_labels)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample_1 = self.mid_block(sample, emb)
        sample_2 = sample_1.clone()

        # 5. up_1, up_2
        skip_sample_1 = None
        skip_sample_2 = None
        for upsample_block_1, upsample_block_2 in zip(self.up_blocks_1, self.up_blocks_2):
            res_samples = down_block_res_samples[-len(upsample_block_1.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block_1.resnets)]

            if hasattr(upsample_block_1, "skip_conv"):
                sample_1, skip_sample_1 = upsample_block_1(sample_1, res_samples, emb, skip_sample_1)
            else:
                sample_1 = upsample_block_1(sample_1, res_samples, emb)

            if hasattr(upsample_block_2, "skip_conv"):
                sample_2, skip_sample_2 = upsample_block_2(sample_2, res_samples, emb, skip_sample_2)
            else:
                sample_2 = upsample_block_2(sample_2, res_samples, emb)

        # 6. post-process_1
        sample_1 = self.conv_norm_out_1(sample_1)
        sample_1 = self.conv_act(sample_1)
        sample_1 = self.conv_out_1(sample_1)

        # 6. post-process_2
        sample_2 = self.conv_norm_out_2(sample_2)
        sample_2 = self.conv_act(sample_2)
        sample_2 = self.conv_out_2(sample_2)

        if skip_sample_1 is not None:
            sample_1 += skip_sample_1

        if skip_sample_2 is not None:
            sample_2 += skip_sample_2

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample_1.shape[0], *([1] * len(sample_1.shape[1:]))))
            sample_1 = sample_1 / timesteps
            sample_2 = sample_2 / timesteps

        if separated_sample:
            return sample_1, sample_2, ratio

        sample = sample_1 * ratio + sample_2 * (1 - ratio)

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)

class YHeadNet2DModel(ModelMixin, ConfigMixin):
    r"""
    YHeadNet2DModel is a 2D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`True`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        num_class_embeds = 10,
        ratio_scheduler_kwargs: dict = {},
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # ratio scheduler
        self.ratio_scheduler = DoubleDenoisingRatioScheduler(**ratio_scheduler_kwargs)

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        self.class_embedding = nn.Linear(num_class_embeds, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out_1 = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        self.conv_out_2 = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        separated_sample: bool = False,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        ratio = self.ratio_scheduler.get_ratio(timesteps, batch=True)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        if class_labels is not None:
            emb += self.class_embedding(class_labels)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample_1 = self.conv_out_1(sample)
        sample_2 = self.conv_out_2(sample)

        if skip_sample is not None:
            sample_1 += skip_sample
            sample_2 += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample_1 = sample_1 / timesteps
            sample_2 = sample_2 / timesteps

        if separated_sample:
            return sample_1, sample_2, ratio
        
        sample = sample_1 * ratio + sample_2 * (1 - ratio)

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)
