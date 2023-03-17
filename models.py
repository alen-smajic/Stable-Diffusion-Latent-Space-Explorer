import os
import random

import diffusers
from torchvision import transforms
import torch
from tqdm.auto import tqdm


class StableDiffusion:
    def __init__(self, model_cfg):
        """
        Stores the model configurations as attributes and loads the Stable Diffusion model instance from the diffusers
        library.

        :param model_cfg: Dictionary containing the model configurations.
        """
        # General parameters
        self.model_identifier = model_cfg["model_identifier"]
        self.exp_identifier = model_cfg["exp_identifier"]
        self.device = torch.device(f"cuda:{model_cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")
        self.sd_pipeline = self._load_sd_pipeline(
            model_cfg["model_id"],
            model_cfg["scheduler"] if "scheduler" in model_cfg else "DPMSolverMultistepScheduler",
            model_cfg["att_slicing"] if "att_slicing" in model_cfg else True,
            model_cfg["vae_slicing"] if "vae_slicing" in model_cfg else True,
            model_cfg["enable_xformers"] if "enable_xformers" in model_cfg else False
        )
        self.diffusion_steps = model_cfg["diffusion_steps"] if "diffusion_steps" in model_cfg else 50
        self.guidance_scale = model_cfg["guidance_scale"] if "guidance_scale" in model_cfg else 7.5

        # Prompt parameters
        self.prompt = model_cfg["prompt"] if "prompt" in model_cfg else None
        self.load_prompt_embeds = model_cfg["load_prompt_embeds"] if "load_prompt_embeds" in model_cfg else None

        # Latent noise, input image and mask parameters
        self.rand_seed = model_cfg["rand_seed"] if "rand_seed" in model_cfg else random.randint(0, 10**6)
        self.images_per_prompt = model_cfg["images_per_prompt"] if "images_per_prompt" in model_cfg else 1
        self.height = model_cfg["height"] if "height" in model_cfg else None
        self.width = model_cfg["width"] if "width" in model_cfg else None
        self.load_latent_noise = model_cfg["load_latent_noise"] if "load_latent_noise" in model_cfg else None
        self.image = model_cfg["image"] if "image" in model_cfg else None
        self.strength = model_cfg["strength"] if "strength" in model_cfg else None
        self.mask = model_cfg["mask"] if "mask" in model_cfg else None

    def _load_sd_pipeline(self, model_id, scheduler_name, att_slicing, vae_slicing, enable_xformers):
        """
        Loads a Stable Diffusion model instance from the diffusers library with the correct configurations.

        :param model_id: String name of the model repository within HuggingFace or the path to the cloned repository,
        containing the model weights and model configurations.
        :param scheduler_name: String name of the scheduler algorithm.
        :param att_slicing: Boolean value specifying whether attention slicing should be used. Reduces memory
        consumption during the diffusion process at the cost of speed.
        :param vae_slicing: Boolean value specifying whether VAE slicing should be used. Reduces memory consumption
        during the encoding/decoding stage at the cost of speed.
        :param enable_xformers: Whether to enable xFormers for optimized performance in the attention blocks (requires
        the xformers package).
        :return: Stable Diffusion model instance from the diffusers library.
        """
        print(f"Loading Stable Diffusion {self.model_identifier} from the diffusers (HuggingFace) library..")

        if self.model_identifier == "txt2img":
            sd_pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            ).to(self.device)
        elif self.model_identifier == "img2img":
            sd_pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            ).to(self.device)
        elif self.model_identifier == "inpaint":
            sd_pipeline = diffusers.StableDiffusionInpaintPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            ).to(self.device)

        sd_pipeline.scheduler = self._load_scheduler(scheduler_name, model_id)

        if att_slicing:
            sd_pipeline.enable_attention_slicing()
        if vae_slicing:
            sd_pipeline.vae.enable_slicing()
        if enable_xformers:
            sd_pipeline.enable_xformers_memory_efficient_attention()

        print(f"Stable Diffusion {self.model_identifier} loaded\n")
        return sd_pipeline

    def _load_scheduler(self, scheduler_name, model_id):
        """
        Loads a scheduler instance from the diffusers library with the correct configurations for Stable Diffusion.

        :param scheduler_name: String name of the scheduler.
        :param model_id: String name of the model repository within HuggingFace or the path to the cloned repository,
        containing the scheduler configurations.
        :return: Scheduler instance.
        """
        if scheduler_name == "EulerDiscreteScheduler":
            scheduler = diffusers.EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "EulerAncestralDiscreteScheduler":
            scheduler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "DDPMScheduler":
            scheduler = diffusers.DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "HeunDiscreteScheduler":
            scheduler = diffusers.HeunDiscreteSchedule.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "DEISMultistepScheduler":
            scheduler = diffusers.DEISMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "DPMSolverMultistepScheduler":
            scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "LMSDiscreteScheduler":
            scheduler = diffusers.LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "DDIMScheduler":
            scheduler = diffusers.DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "KDPM2AncestralDiscreteScheduler":
            scheduler = diffusers.KDPM2AncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "DPMSolverSinglestepScheduler":
            scheduler = diffusers.DPMSolverSinglestepScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "PNDMScheduler":
            scheduler = diffusers.PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "KDPM2DiscreteScheduler":
            scheduler = diffusers.KDPM2DiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        if scheduler_name == "UniPCMultistepScheduler":
            scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        print("Scheduler loaded")
        return scheduler

    def sample_noise(self, height, width, images_per_prompt, rand_seed):
        """
        Samples a new latent noise tensor.

        :param height: Image height of the desired VAE output (used for computing the latent noise height).
        :param width: Image width of the desired VAE output (used for computing the latent noise width).
        :param images_per_prompt: Amount of images to generate per prompt (specifies the batch dimension of the latent
        noise).
        :param rand_seed: Random seed for sampling reproducible random noise.
        :return: A torch tensor of the latent noise.
        """
        scale_factor = 2 ** (len(self.sd_pipeline.vae.config.block_out_channels) - 1)
        shape = (images_per_prompt, self.sd_pipeline.vae.latent_channels, height // scale_factor, width // scale_factor)
        latent_noise = diffusers.utils.randn_tensor(
            shape,
            generator=torch.Generator("cpu").manual_seed(rand_seed),
            device=self.device,
            dtype=torch.float16
        )

        return latent_noise

    @torch.no_grad()
    def encode_prompt(self, prompt):
        """
        Generates the prompt embedding by encoding the given prompt through a text encoder.

        :param prompt: Input prompt where the positive part is separated from the negative part by a vertical line "|"
        without any whitespace in between.
        :return: Encoded negative and positive prompt embeddings stacked into a single tensor (batch dimension 2).
        """
        pos_prompt = prompt.split("|")[0]
        neg_prompt = prompt.split("|")[1]
        prompt_embed = self.sd_pipeline._encode_prompt(
            prompt=pos_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg_prompt if neg_prompt else None
        )
        return prompt_embed

    @torch.no_grad()
    def encode_images(self, images):
        """
        Encodes a torch tensor of images using the VAE from Stable Diffusion.

        :param images: Torch tensor of images with normalized values between -1 and 1.
        :return: Torch tensor of the encoded image embeddings.
        """
        generator = torch.Generator("cpu").manual_seed(self.rand_seed)
        image_embeds = self.sd_pipeline.vae.encode(images).latent_dist.sample(generator)
        return image_embeds

    @torch.no_grad()
    def decode_images(self, images):
        """
        Decodes a torch tensor of latent images using the VAE from Stable Diffusion.

        :param images: Torch tensor of encoded latent images.
        :return: List of decoded PIL images.
        """
        transform = transforms.ToPILImage()
        images = self.sd_pipeline.vae.decode(images).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        pil_imgs = []
        for batch_idx in range(images.shape[0]):
            pil_imgs.append(transform(images[batch_idx].float()))
        return pil_imgs

    def load_noise(self, load_latent_noise, height, width, images_per_prompt, rand_seed):
        """
        Loads the latent random gaussian noise tensor by either loading it from a local file or by sampling new noise.

        :param load_latent_noise: Path to a local file containing the latent noise tensor.
        :param height: Image height of the desired VAE output (used for computing the latent noise height).
        :param width: Image width of the desired VAE output (used for computing the latent noise width).
        :param images_per_prompt: Amount of images to generate per prompt (specifies the batch dimension of the latent
        noise).
        :param rand_seed: Random seed for sampling reproducible random noise.
        :return: A torch tensor of the latent noise.
        """
        if os.path.isfile(load_latent_noise):
            print(f"Loading latent noise from {load_latent_noise}")
            latent_noise = torch.load(load_latent_noise)["latent_noise"].to(self.device)
            # Adjusts the height and width attributes with respect to the loaded latent noise tensor
            scale_factor = 2 ** (len(self.sd_pipeline.vae.config.block_out_channels) - 1)
            self.height = latent_noise.shape[2] * scale_factor
            self.width = latent_noise.shape[3] * scale_factor
            self.images_per_prompt = 1
            return latent_noise
        else:
            return self.sample_noise(height, width, images_per_prompt, rand_seed)

    def load_prompt(self, load_prompt_embeds, prompt):
        """
        Loads the prompt embedding by either loading it from a local file or by encoding a given string prompt.

        :param load_prompt_embeds: Path to a local file containing the prompt embeddings.
        :param prompt: Input prompt where the positive part is separated from the negative part by a vertical line "|"
        without any whitespace in between.
        :return: Encoded negative and positive prompt embeddings stacked into a single tensor (batch dimension 2).
        """
        if os.path.isfile(load_prompt_embeds):
            print(f"Loading prompt embeddings from {load_prompt_embeds}")
            prompt_embed = torch.load(load_prompt_embeds)["prompt_embed"].to(self.device)
            return prompt_embed
        else:
            return self.encode_prompt(prompt)


class Txt2Img(StableDiffusion):
    def __init__(self, model_cfg):
        super(Txt2Img, self).__init__(model_cfg)

    @ torch.no_grad()
    def run_sd_inference(self, prompt_embed, latent_noise, image=None, mask=None, visualize_diffusion=False):
        """
        Runs the Stable Diffusion inference process for a given encoded prompt embedding and a latent noise tensor.

        :param prompt_embed: Torch tensor of an encoded prompt embedding.
        :param latent_noise: Latent random gaussian noise tensor.
        :param visualize_diffusion: Whether to decode the Stable Diffusion output after each diffusion step in order
        to visualize the diffusion process.
        :return: A list containing the VAE image embeddings produced by Stable Diffusion and a list of images
        produced by decoding the VAE image embeddings.
        """
        # Sets the diffusion steps
        self.sd_pipeline.scheduler.set_timesteps(self.diffusion_steps, device=self.device)

        # Scales the latent noise by the standard deviation required by the scheduler
        latent_noise *= self.sd_pipeline.scheduler.init_noise_sigma

        # Reshapes the prompt embeddings tensor to be consistent with the latent noise batch size (images per prompt)
        bs_embed, seq_len, _ = prompt_embed.shape
        # Duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embed = prompt_embed.repeat(1, latent_noise.shape[0], 1)
        prompt_embed = prompt_embed.view(bs_embed * latent_noise.shape[0], seq_len, -1)

        # Diffusion loop
        results = [[] for _ in range(latent_noise.shape[0])]  # Stores the results of each diffusion step
        for t in tqdm(self.sd_pipeline.scheduler.timesteps):
            # Expand the latents to avoid doing two forward passes for classifier-free guidance
            latent_model_input = torch.cat([latent_noise] * 2)
            latent_model_input = self.sd_pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.sd_pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embed).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous latent noise sample x_t -> x_t-1
            latent_noise = self.sd_pipeline.scheduler.step(noise_pred, t, latent_noise).prev_sample

            # Stores each generated latent noise tensor
            for i in range(latent_noise.shape[0]):
                results[i].append(latent_noise[i].unsqueeze(0))

        print("Decoding images..\n")
        img_embeds = [[] for _ in range(len(results))]  # Placeholder for all processed image embeddings
        images = [[] for _ in range(len(results))]  # Placeholder for all decoded image embeddings
        for batch_idx in range(len(results)):
            for diffusion_step in range(len(results[i])):
                if not visualize_diffusion and diffusion_step != len(results[i])-1:
                    # Skip all diffusion steps except the last one
                    continue
                img_embeds[batch_idx].append(
                    results[batch_idx][diffusion_step] / self.sd_pipeline.vae.config.scaling_factor
                )
                images[batch_idx].append(self.decode_images(img_embeds[batch_idx][-1])[0])

        return img_embeds, images


class Img2Img(StableDiffusion):
    def __init__(self, model_cfg):
        super(Img2Img, self).__init__(model_cfg)

    @ torch.no_grad()
    def run_sd_inference(self, prompt_embed, latent_noise, image, mask=None, visualize_diffusion=False):
        """
        Runs the Stable Diffusion inference process for a given encoded prompt embedding and an input image.

        :param prompt_embed: Torch tensor of an encoded prompt embedding.
        :param latent_noise: Latent random gaussian noise tensor.
        :param image: Input image, which is being edited by the Stable Diffusion img2img model.
        :param visualize_diffusion: Whether to decode the Stable Diffusion output after each diffusion step in order
        to visualize the diffusion process.
        :return: A list containing the VAE image embeddings produced by Stable Diffusion and a list of images
        produced by decoding the VAE image embeddings.
        """
        # Sets the diffusion steps
        self.sd_pipeline.scheduler.set_timesteps(self.diffusion_steps, device=self.device)
        # Uses the strength to scale the amount of diffusion steps and the starting point
        timesteps, num_diffusion_steps = self.sd_pipeline.get_timesteps(self.diffusion_steps, self.strength, self.device)
        latent_timestep = timesteps[:1].repeat(self.images_per_prompt)

        # Processes the input image and encodes it into the latent space of the VAE
        image = image.resize((self.width, self.height))
        image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess(image)
        image = image.to(device=self.device, dtype=torch.float16)
        encoded_img = self.encode_images(image)
        encoded_img *= self.sd_pipeline.vae.config.scaling_factor
        # Stacks the encoded images to be consistent with the batch size (images per prompt)
        additional_image_per_prompt = self.images_per_prompt // encoded_img.shape[0]
        encoded_imgs = torch.cat([encoded_img] * additional_image_per_prompt, dim=0)
        # Applies the latent noise tensor to the encoded images
        latent_noise = self.sd_pipeline.scheduler.add_noise(encoded_imgs, latent_noise, latent_timestep)

        # Reshapes the prompt embeddings tensor to be consistent with the latent noise batch size (images per prompt)
        bs_embed, seq_len, _ = prompt_embed.shape
        # Duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embed = prompt_embed.repeat(1, latent_noise.shape[0], 1)
        prompt_embed = prompt_embed.view(bs_embed * latent_noise.shape[0], seq_len, -1)

        # Diffusion loop
        results = [[] for _ in range(latent_noise.shape[0])]  # Stores the results of each diffusion step
        for t in tqdm(timesteps):
            # Expand the latents to avoid doing two forward passes for classifier-free guidance
            latent_model_input = torch.cat([latent_noise] * 2)
            latent_model_input = self.sd_pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.sd_pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embed).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous latent noise sample x_t -> x_t-1
            latent_noise = self.sd_pipeline.scheduler.step(noise_pred, t, latent_noise).prev_sample

            # Stores each generated latent noise tensor
            for i in range(latent_noise.shape[0]):
                results[i].append(latent_noise[i].unsqueeze(0))

        print("Decoding images..\n")
        img_embeds = [[] for _ in range(len(results))]  # Placeholder for all processed image embeddings
        images = [[] for _ in range(len(results))]  # Placeholder for all decoded image embeddings
        for batch_idx in range(len(results)):
            for diffustion_step in range(len(results[i])):
                if not visualize_diffusion and diffustion_step != len(results[i])-1:
                    # Skip all diffusion steps except the last one
                    continue
                img_embeds[batch_idx].append(
                    results[batch_idx][diffustion_step] / self.sd_pipeline.vae.config.scaling_factor
                )
                images[batch_idx].append(self.decode_images(img_embeds[batch_idx][-1])[0])

        return img_embeds, images


class Inpaint(StableDiffusion):
    def __init__(self, model_cfg):
        super(Inpaint, self).__init__(model_cfg)

    @ torch.no_grad()
    def run_sd_inference(self, prompt_embed, latent_noise, image, mask, visualize_diffusion=False):
        """
        Runs the Stable Diffusion inference process for a given encoded prompt embedding, an input image and a mask.

        :param prompt_embed: Torch tensor of an encoded prompt embedding.
        :param latent_noise: Latent random gaussian noise tensor.
        :param image: Input image, which is being edited by the Stable Diffusion inpaint model.
        :param mask: Input mask, which is used to constrain the inpainting by the Stable Diffusion inpaint model.
        :param visualize_diffusion: Whether to decode the Stable Diffusion output after each diffusion step in order
        to visualize the diffusion process.
        :return: A list containing the VAE image embeddings produced by Stable Diffusion and a list of images
        produced by decoding the VAE image embeddings.
        """
        # Sets the diffusion steps
        self.sd_pipeline.scheduler.set_timesteps(self.diffusion_steps, device=self.device)

        # Scales the latent noise by the standard deviation required by the scheduler
        latent_noise *= self.sd_pipeline.scheduler.init_noise_sigma

        # Processes the input image and mask, and encodes it into the latent space of the VAE
        image = image.resize((self.width, self.height))
        mask = mask.resize((self.width, self.height))
        mask, masked_image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.\
            prepare_mask_and_masked_image(image, mask)
        mask, masked_image_latents = self.sd_pipeline.prepare_mask_latents(
            mask=mask,
            masked_image=masked_image,
            batch_size=latent_noise.shape[0],
            height=self.height,
            width=self.width,
            dtype=prompt_embed.dtype,
            device=self.device,
            generator=torch.Generator("cpu").manual_seed(self.rand_seed),
            do_classifier_free_guidance=True
        )

        # Reshapes the prompt embeddings tensor to be consistent with the latent noise batch size (images per prompt)
        bs_embed, seq_len, _ = prompt_embed.shape
        # Duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embed = prompt_embed.repeat(1, latent_noise.shape[0], 1)
        prompt_embed = prompt_embed.view(bs_embed * latent_noise.shape[0], seq_len, -1)

        # Diffusion loop
        results = [[] for _ in range(latent_noise.shape[0])]  # Stores the results of each diffusion step
        for t in tqdm(self.sd_pipeline.scheduler.timesteps):
            # Expand the latents to avoid doing two forward passes for classifier-free guidance
            latent_model_input = torch.cat([latent_noise] * 2)
            latent_model_input = self.sd_pipeline.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            noise_pred = self.sd_pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embed).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous latent noise sample x_t -> x_t-1
            latent_noise = self.sd_pipeline.scheduler.step(noise_pred, t, latent_noise).prev_sample

            # Stores each generated latent noise tensor
            for i in range(latent_noise.shape[0]):
                results[i].append(latent_noise[i].unsqueeze(0))

        print("Decoding images..\n")
        img_embeds = [[] for _ in range(len(results))]  # Placeholder for all processed image embeddings
        images = [[] for _ in range(len(results))]  # Placeholder for all decoded image embeddings
        for batch_idx in range(len(results)):
            for diffustion_step in range(len(results[i])):
                if not visualize_diffusion and diffustion_step != len(results[i])-1:
                    # Skip all diffusion steps except the last one
                    continue
                img_embeds[batch_idx].append(
                    results[batch_idx][diffustion_step] / self.sd_pipeline.vae.config.scaling_factor
                )
                images[batch_idx].append(self.decode_images(img_embeds[batch_idx][-1])[0])

        return img_embeds, images