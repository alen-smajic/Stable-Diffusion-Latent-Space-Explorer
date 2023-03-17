import random

from PIL import Image
import torch

import utils


def run_single_inference(cfg_path, exp_cfg, sd_model):
    """
    Runs a simple single inference experiment for the given Stable Diffusion model.

    :param cfg_path: Path to the experiment configuration file.
    :param exp_cfg: Dictionary containing the experiment configuration.
    :param sd_model: The Stable Diffusion model instance.
    """
    # Creates the output folder for storing the experiment results
    output_path = utils.generate_output_folder(
        exp_cfg["model_identifier"],
        exp_cfg["exp_identifier"],
        cfg_path,
        exp_cfg["output_path"],
        gifs=False
    )

    # Loads the inputs for Stable Diffusion (prompt embedding and latent noise)
    prompt_embed = sd_model.load_prompt(exp_cfg["load_prompt_embeds"], exp_cfg["prompt"])
    latent_noise = sd_model.load_noise(
        exp_cfg["load_latent_noise"],
        exp_cfg["height"],
        exp_cfg["width"],
        exp_cfg["images_per_prompt"],
        exp_cfg["rand_seed"]
    )

    # Loads the input image and mask if supported by the model
    if exp_cfg["model_identifier"] == "txt2img":
        image, mask = None, None
    elif exp_cfg["model_identifier"] == "img2img":
        image, mask = Image.open(exp_cfg["image"]).convert("RGB"), None
    elif exp_cfg["model_identifier"] == "inpaint":
        image = Image.open(exp_cfg["image"]).convert("RGB")
        mask = Image.open(exp_cfg["mask"]).convert("RGB")

    # Runs the inference process for Stable Diffusion
    image_embeds, images = sd_model.run_sd_inference(prompt_embed, latent_noise, image, mask)

    # Stores the experiment results
    for batch_idx in range(len(images)):
        utils.save_sd_results(
            output_path=output_path,
            prompt_emb=prompt_embed.cpu(),
            latent_noise=latent_noise[batch_idx].unsqueeze(0).cpu(),
            image_embed=image_embeds[batch_idx][-1].cpu(),
            image=images[batch_idx][-1],
            file_name=f"output-{batch_idx}_diffstep-{sd_model.diffusion_steps}"
        )

    print("Experiment finished")


def run_visualize_diffusion(cfg_path, exp_cfg, sd_model):
    """
    Runs a simple single inference experiment for the given Stable Diffusion model and visualizes each diffusion step
    by decoding the latents after each iteration.

    :param cfg_path: Path to the experiment configuration file.
    :param exp_cfg: Dictionary containing the experiment configuration.
    :param sd_model: The Stable Diffusion model instance.
    """
    # Creates the output folder for storing the experiment results
    output_path = utils.generate_output_folder(
        exp_cfg["model_identifier"],
        exp_cfg["exp_identifier"],
        cfg_path,
        exp_cfg["output_path"]
    )

    # Loads the inputs for Stable Diffusion (prompt embedding and latent noise)
    prompt_embed = sd_model.load_prompt(exp_cfg["load_prompt_embeds"], exp_cfg["prompt"])
    latent_noise = sd_model.load_noise(
        exp_cfg["load_latent_noise"],
        exp_cfg["height"],
        exp_cfg["width"],
        exp_cfg["images_per_prompt"],
        exp_cfg["rand_seed"]
    )

    # Loads the input image and mask if supported by the model
    if exp_cfg["model_identifier"] == "txt2img":
        image, mask = None, None
    elif exp_cfg["model_identifier"] == "img2img":
        image, mask = Image.open(exp_cfg["image"]).convert("RGB"), None
    elif exp_cfg["model_identifier"] == "inpaint":
        image = Image.open(exp_cfg["image"]).convert("RGB")
        mask = Image.open(exp_cfg["mask"]).convert("RGB")

    # Runs the inference process for Stable Diffusion
    image_embeds, images = sd_model.run_sd_inference(prompt_embed, latent_noise, image, mask, visualize_diffusion=True)

    # Stores the experiment results
    for batch_idx in range(len(images)):
        for diff_step in range(len(images[batch_idx])):
            utils.save_sd_results(
                output_path=output_path,
                prompt_emb=prompt_embed.cpu(),
                latent_noise=latent_noise[batch_idx].unsqueeze(0).cpu(),
                image_embed=image_embeds[batch_idx][diff_step].cpu(),
                image=images[batch_idx][diff_step],
                file_name=f"output-{batch_idx}_diffstep-{diff_step}"
            )

    # Produces a gif to visualize each diffusion step
    utils.produce_gif(images, output_path, exp_cfg["gif_frame_dur"])

    print("Experiment finished")


def run_random_walk(cfg_path, exp_cfg, sd_model):
    """
    Runs a random walk experiment for the given Stable Diffusion model.

    :param cfg_path: Path to the experiment configuration file.
    :param exp_cfg: Dictionary containing the experiment configuration.
    :param sd_model: The Stable Diffusion model instance.
    """
    # Creates the output folder for storing the experiment results
    output_path = utils.generate_output_folder(
        exp_cfg["model_identifier"],
        exp_cfg["exp_identifier"],
        cfg_path,
        exp_cfg["output_path"]
    )

    # Loads the inputs for Stable Diffusion (prompt embedding and latent noise)
    prompt_embed = sd_model.load_prompt(exp_cfg["load_prompt_embeds"], exp_cfg["prompt"])
    latent_noise = sd_model.load_noise(
        exp_cfg["load_latent_noise"],
        exp_cfg["height"],
        exp_cfg["width"],
        exp_cfg["images_per_prompt"],
        exp_cfg["rand_seed"]
    )

    # Loads the input image and mask if supported by the model
    if exp_cfg["model_identifier"] == "txt2img":
        image, mask = None, None
    elif exp_cfg["model_identifier"] == "img2img":
        image, mask = Image.open(exp_cfg["image"]).convert("RGB"), None
    elif exp_cfg["model_identifier"] == "inpaint":
        image = Image.open(exp_cfg["image"]).convert("RGB")
        mask = Image.open(exp_cfg["mask"]).convert("RGB")

    results = [[] for _ in range(latent_noise.shape[0])]
    for direction in range(exp_cfg["walk_directions"]):
        # Initial prompt embedding and latent noise
        prompt_emb = prompt_embed.clone()
        lat_noise = latent_noise.clone()

        # Randomly chosen noise and prompt deltas for the random walk in a specific direction
        noise_delta = exp_cfg["step_size"] * torch.empty_like(lat_noise).uniform_(-1, 1)
        prompt_delta = exp_cfg["step_size"] * torch.empty_like(prompt_emb).uniform_(-1, 1)

        for step in range(exp_cfg["walk_steps"] + 1):  # Step 0 is the initial image
            if step == 0:
                step = "start"

            print(f"Random walk direction {direction+1} of {exp_cfg['walk_directions']} at step {step} of "
                  f"{exp_cfg['walk_steps']}")

            if exp_cfg["prompt_rand_walk"] and step != "start":
                prompt_emb += prompt_delta

            if exp_cfg["noise_rand_walk"] and step != "start":
                lat_noise += noise_delta

            # Runs the inference process for Stable Diffusion
            image_embeds, images = sd_model.run_sd_inference(prompt_emb, lat_noise, image, mask)

            # Stores the experiment results
            for batch_idx in range(len(images)):
                utils.save_sd_results(
                    output_path=output_path,
                    prompt_emb=prompt_emb.cpu(),
                    latent_noise=lat_noise[batch_idx].unsqueeze(0).cpu(),
                    image_embed=image_embeds[batch_idx][-1].cpu(),
                    image=images[batch_idx][-1],
                    file_name=f"output-{batch_idx}_direction-{direction}_randwalkstep-{step}"
                )
                results[batch_idx].append(images[batch_idx][-1])

        # Adds all generated images from the single direction random walk in the opposite order to produce a rubber-band
        # gif (walks the way back to the initial image)
        for batch_idx in range(len(results)):
            tmp = results[batch_idx][(2 * direction * (exp_cfg["walk_steps"] + 1)):]
            results[batch_idx] += tmp[::-1]

    # Produces a gif to visualize each random walk step
    utils.produce_gif(results, output_path, exp_cfg["gif_frame_dur"])

    print("Experiment finished")


def run_interpolation(cfg_path, exp_cfg, sd_model):
    """
    Runs an interpolation experiment for the given Stable Diffusion model.

    :param cfg_path: Path to the experiment configuration file.
    :param exp_cfg: Dictionary containing the experiment configuration.
    :param sd_model: The Stable Diffusion model instance.
    """
    # Creates the output folder for storing the experiment results
    output_path = utils.generate_output_folder(
        exp_cfg["model_identifier"],
        exp_cfg["exp_identifier"],
        cfg_path,
        exp_cfg["output_path"]
    )

    # Loads the prompt embedding
    if len(exp_cfg["inter_prompts"]) > 1:  # Interpolates between multiple prompt embeddings
        print(f"Interpolating {len(exp_cfg['inter_prompts'])} prompts.")
        log_txt = "prompt"
        prompt_embeds = []
        for i in range(len(exp_cfg["inter_prompts"]) - 1):
            prompt_embed_1 = sd_model.load_prompt(
                load_prompt_embeds=exp_cfg["inter_prompts"][i],
                prompt=exp_cfg["inter_prompts"][i]
            )
            prompt_embed_2 = sd_model.load_prompt(
                load_prompt_embeds=exp_cfg["inter_prompts"][i + 1],
                prompt=exp_cfg["inter_prompts"][i + 1]
            )
            prompt_embeds += utils.interpolate(
                x=prompt_embed_1,
                y=prompt_embed_2,
                steps=exp_cfg["interpolation_steps"],
                interpolation_method=exp_cfg["interpolation_method"]
            )
        prompt_stepper = 1
    else:  # Loads a single prompt embedding
        log_txt = ""
        prompt_embed = sd_model.load_prompt(
            load_prompt_embeds=exp_cfg["inter_prompts"][0],
            prompt=exp_cfg["inter_prompts"][0]
        )
        prompt_embeds = [prompt_embed]
        prompt_stepper = 0

    # Loads the latent noise
    if len(exp_cfg["inter_noises"]) > 1:  # Interpolates between multiple latent noise embeddings
        print(f"Interpolating {len(exp_cfg['inter_noises'])} latent gaussian noise tensors.")
        log_txt += " and latent noise" if log_txt else "latent noise"
        latent_noise = []
        for i in range(len(exp_cfg["inter_noises"]) - 1):
            sd_model.rand_seed = exp_cfg["inter_noises"][i]
            noise_1 = sd_model.load_noise(
                load_latent_noise=exp_cfg["inter_noises"][i],
                height=exp_cfg["height"],
                width=exp_cfg["width"],
                images_per_prompt=1,
                rand_seed=exp_cfg["inter_noises"][i]
            )
            sd_model.rand_seed = exp_cfg["inter_noises"][i + 1]
            noise_2 = sd_model.load_noise(
                load_latent_noise=exp_cfg["inter_noises"][i + 1],
                height=exp_cfg["height"],
                width=exp_cfg["width"],
                images_per_prompt=1,
                rand_seed=exp_cfg["inter_noises"][i]
            )
            latent_noise += utils.interpolate(
                x=noise_1,
                y=noise_2,
                steps=exp_cfg["interpolation_steps"],
                interpolation_method=exp_cfg["interpolation_method"]
            )
        noise_stepper = 1
    else:  # Loads a single latent noise tensor
        latent_noise = sd_model.load_noise(
            load_latent_noise=exp_cfg["inter_noises"][0],
            height=exp_cfg["height"],
            width=exp_cfg["width"],
            images_per_prompt=1,
            rand_seed=exp_cfg["inter_noises"][0]
        )
        latent_noise = [latent_noise]
        noise_stepper = 0

    # Loads the input image and mask if supported by the model
    if exp_cfg["model_identifier"] == "txt2img":
        image, mask = None, None
    elif exp_cfg["model_identifier"] == "img2img":
        image, mask = Image.open(exp_cfg["image"]).convert("RGB"), None
    elif exp_cfg["model_identifier"] == "inpaint":
        image = Image.open(exp_cfg["image"]).convert("RGB")
        mask = Image.open(exp_cfg["mask"]).convert("RGB")

    # Iterates over all interpolation steps
    results = [[]]  # Interpolation only supports a batch size of 1
    for int_idx in range(max(len(prompt_embeds), len(latent_noise))):
        lst_itm = int_idx // (exp_cfg["interpolation_steps"] + 2)
        step = int_idx % (exp_cfg["interpolation_steps"] + 2)
        if step == 0:
            step = "start"
        elif step == exp_cfg["interpolation_steps"] + 1:
            step = "end"
        print(f"Interpolating {log_txt} list items at index {lst_itm} and {lst_itm+1} at step {step} of "
              f"{exp_cfg['interpolation_steps']}")

        # Runs the inference process for Stable Diffusion
        image_embeds, images = sd_model.run_sd_inference(
            prompt_embeds[int_idx * prompt_stepper],
            latent_noise[int_idx * noise_stepper],
            image,
            mask
        )

        # Stores the experiment results
        for batch_idx in range(len(images)):
            utils.save_sd_results(
                output_path=output_path,
                prompt_emb=prompt_embeds[int_idx * prompt_stepper].cpu(),
                latent_noise=latent_noise[int_idx * noise_stepper].cpu(),
                image_embed=image_embeds[batch_idx][-1].cpu(),
                image=images[batch_idx][-1],
                file_name=f"output-{batch_idx}_lstitms-{lst_itm},{lst_itm + 1}_step-{step}"
            )
            results[batch_idx].append(images[batch_idx][-1])

    # Produces a gif to visualize each interpolation step
    utils.produce_gif(results, output_path, exp_cfg["gif_frame_dur"])

    print("Experiment finished")


def run_diffevolution(cfg_path, exp_cfg, sd_model):
    # Creates the output folder for storing the experiment results
    output_path = utils.generate_output_folder(
        exp_cfg["model_identifier"],
        exp_cfg["exp_identifier"],
        cfg_path,
        exp_cfg["output_path"],
        gifs=False
    )

    # Loads the initial inputs for Stable Diffusion (prompt embedding and latent noise)
    prompt_embed = sd_model.load_prompt(exp_cfg["load_prompt_embeds"], exp_cfg["prompt"])
    latent_noise = sd_model.load_noise(
        exp_cfg["load_latent_noise"],
        exp_cfg["height"],
        exp_cfg["width"],
        1,
        exp_cfg["rand_seed"]
    )

    # Loads the input image and mask if supported by the model
    if exp_cfg["model_identifier"] == "txt2img":
        image, mask = None, None
    elif exp_cfg["model_identifier"] == "img2img":
        image, mask = Image.open(exp_cfg["image"]).convert("RGB"), None
    elif exp_cfg["model_identifier"] == "inpaint":
        image = Image.open(exp_cfg["image"]).convert("RGB")
        mask = Image.open(exp_cfg["mask"]).convert("RGB")

    print("Creating initial latents..")
    # Runs the inference process for Stable Diffusion
    image_embeds, images = sd_model.run_sd_inference(prompt_embed, latent_noise, image, mask)
    # Stores the generated initial image and latents
    utils.save_sd_results(
        output_path=output_path,
        prompt_emb=prompt_embed.cpu(),
        latent_noise=latent_noise[0].unsqueeze(0).cpu(),
        image_embed=image_embeds[0][-1].cpu(),
        image=images[0][-1],
        file_name=f"difevolstep-0"
    )

    step = 1  # Tracks the amount of diffevolution steps
    choice = 0  # Tracks the index of the most dominant gene
    while True:  # Performs diffevolution
        print(f"Diffevolution step: {step}")

        # Stacks the latents
        latent_noise = latent_noise.repeat(latent_noise.shape[0], 1, 1, 1)

        # Samples several latent noise tensor as new genes
        distant = sd_model.load_noise(
            "None",
            exp_cfg["height"],
            exp_cfg["width"],
            exp_cfg["genes_per_generation"],
            random.randint(0, 10**6)
        )

        # Transfers some of the new latent code features to the current latent noise
        new_gen_latents = utils.slerp(
            latent_noise,
            distant,
            exp_cfg["step_size"],
        )

        # Runs the inference process for Stable Diffusion
        image_embeds, images = sd_model.run_sd_inference(prompt_embed, new_gen_latents, image, mask)

        # Stores the experiment results
        for batch_idx in range(len(images)):
            utils.save_sd_results(
                output_path=output_path,
                prompt_emb=prompt_embed.cpu(),
                latent_noise=new_gen_latents[batch_idx].unsqueeze(0).cpu(),
                image_embed=image_embeds[batch_idx][-1].cpu(),
                image=images[batch_idx][-1],
                file_name=f"difevolstep-{step}_parent-{choice}_gene-{batch_idx}"
            )

        user_input = input(f"Enter stop and press enter to stop diffevolution.\n"
                           f"Choose the dominant gene (0-{exp_cfg['genes_per_generation']}) or "
                       f"leave empty for a re-roll and press enter:  ")
        print("")
        if not user_input.isnumeric():
            print("re-rolling..\n")
            latent_noise = latent_noise[0].unsqueeze(0)
            continue
        elif user_input == "exit":
            break
        else:
            choice = user_input
            latent_noise = new_gen_latents[int(choice)].unsqueeze(0)
            step += 1

    print("Experiment finished")


def a():
    image_size = (512, 512)
    zoom_speed = 128
    num_filler_frames = 64
    resize_factor = 1