import random

import numpy as np
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
            noise_1 = sd_model.load_noise(
                load_latent_noise=exp_cfg["inter_noises"][i],
                height=exp_cfg["height"],
                width=exp_cfg["width"],
                images_per_prompt=1,
                rand_seed=exp_cfg["inter_noises"][i]
            )
            noise_2 = sd_model.load_noise(
                load_latent_noise=exp_cfg["inter_noises"][i + 1],
                height=exp_cfg["height"],
                width=exp_cfg["width"],
                images_per_prompt=1,
                rand_seed=exp_cfg["inter_noises"][i + 1]
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
                file_name=f"interpolation_lstitms-{lst_itm},{lst_itm + 1}_step-{step}"
            )
            results[batch_idx].append(images[batch_idx][-1])

    # Produces a gif to visualize each interpolation step
    utils.produce_gif(results, output_path, exp_cfg["gif_frame_dur"])

    print("Experiment finished")


def run_diffevolution(cfg_path, exp_cfg, sd_model):
    """
    Runs a diffevolution experiment for the given Stable Diffusion model.

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

    print("Creating initial latents..")
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

    # Runs the inference process for Stable Diffusion
    image_embeds, images = sd_model.run_sd_inference(prompt_embed, latent_noise, image, mask)
    # Stores the generated initial image and latents
    utils.save_sd_results(
        output_path=output_path,
        prompt_emb=prompt_embed.cpu(),
        latent_noise=latent_noise[0].unsqueeze(0).cpu(),
        image_embed=image_embeds[0][-1].cpu(),
        image=images[0][-1],
        file_name=f"difevostep-start"
    )

    results = [images[0][-1]]  # Used for storing the gif frames
    step = 1  # Tracks the current diffevolution step
    dom_gene_idx = "start"  # Tracks the index of the most dominant gene
    distant_prompt = prompt_embed  # Used for storing the prompt embeddings of modified prompts
    steps_to_skip = 0  # Used for tracking the amount of diffevolution steps for skipping the user input window
    while True:  # Performs diffevolution
        print(f"Diffevolution step: {step}")

        # Stacks the latents
        latent_noise = latent_noise.repeat(latent_noise.shape[0], 1, 1, 1)

        # Samples several latent noise tensors as new genes
        distant_noise = sd_model.load_noise(
            load_latent_noise="None",
            height=exp_cfg["height"],
            width=exp_cfg["width"],
            images_per_prompt=exp_cfg["genes_per_generation"],
            rand_seed=random.randint(0, 10**6)
        )

        # Transfers some of the new latent code and prompt features to the current latent noise and prompt embeddings
        new_gen_latents = utils.slerp(
            latent_noise,
            distant_noise,
            exp_cfg["step_size"]
        )
        new_gen_prompt = utils.slerp(
            prompt_embed,
            distant_prompt,
            exp_cfg["step_size"]
        )

        # Runs the inference process for Stable Diffusion
        image_embeds, images = sd_model.run_sd_inference(new_gen_prompt, new_gen_latents, image, mask)

        # Stores the experiment results
        for batch_idx in range(len(images)):
            utils.save_sd_results(
                output_path=output_path,
                prompt_emb=new_gen_prompt.cpu(),
                latent_noise=new_gen_latents[batch_idx].unsqueeze(0).cpu(),
                image_embed=image_embeds[batch_idx][-1].cpu(),
                image=images[batch_idx][-1],
                file_name=f"difevostep-{step}_parent-{dom_gene_idx}_gene-{batch_idx}"
            )

        while True:
            if steps_to_skip == 0:
                print(" * ACTION REQUIRED * ")
                print("-> Type exit and press enter to stop the experiment.")
                print("")
                print("-> Press enter (without any input) to re-roll the current generation with the same parameters.")
                print("")
                print("-> Otherwise, please specify a valid action.")
                print("       Valid actions have the form: {int_1};{int_2};{prmpt}")
                print("       {int_1} is the index of the most dominant gene from the current generation.")
                print(f"       Specify a number between 0 and {exp_cfg['genes_per_generation']}.")
                print("       {int_2} is optional and can be used to specify the amount of steps that should perform "
                      "automatically (skips this input window for that amount of steps and randomly chooses genes).")
                print("       {prmpt} is optional and can be used to specify a new prompt that should further guide the "
                      "diffevolution process (use | to separate the positive and negative part of the prompt).")
                print("")
                print("Example input: 2;;")
                print("Selects the gene-2 image of the current generation. The other two parameters remain unspecified.")
                print("")
                user_action = input("Input: ")
                print("")
            else:
                steps_to_skip -= 1
                user_action = False
                break

            if user_action == "exit" or user_action == "Exit":
                print("\nGenerating the final gif..")
                # Produces a gif to visualize the diffevolution process
                utils.produce_gif([results], output_path, exp_cfg["gif_frame_dur"])
                print("Experiment finished")
                exit()

            if user_action == "" or user_action.count(";") == 2:
                break
            else:
                print("\nIt seems like your input could not be recognized. Please specify a valid input.\n")
                continue

        if user_action == "":
            print("\nRe-rolling the current generation")
            continue  # Re-rolls the current generation
        elif user_action is False:
            user_action = f"{random.randint(0, exp_cfg['genes_per_generation']-1)};;"

        user_actions = user_action.split(";")
        if user_actions[0].isnumeric():
            # Loads the most dominant gene
            dom_gene_idx = int(user_actions[0])
            latent_noise = new_gen_latents[dom_gene_idx].unsqueeze(0)
            results.append(images[dom_gene_idx][-1])
            prompt_embed = new_gen_prompt
            step += 1
        if user_actions[1].isnumeric():
            # Loads the amount of steps to skip the user input window
            steps_to_skip = int(user_actions[1])
        if "|" in user_actions[2]:
            # Loads the new prompt
            distant_prompt = sd_model.load_prompt(load_prompt_embeds="None", prompt=user_actions[2])


def run_outpaint_walk(cfg_path, exp_cfg, sd_model):
    """
    Runs an outpaint walk experiment for the given Stable Diffusion model.

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

    # Loads the initial image
    curr_img = Image.open(exp_cfg["image"]).convert("RGB")
    curr_img = curr_img.resize((exp_cfg["width"], exp_cfg["height"]))
    curr_img.save(f"{output_path}/images/init.png")

    # Loads and encodes all prompts
    prompt_embeds = []
    for prmpt in exp_cfg["prompts"]:
        prompt_embeds.append(sd_model.load_prompt(prmpt, prmpt))

    # Used for tracking the current prompt and camera action
    curr_prmpt = prompt_embeds[0]
    curr_prmpt_idx = 0
    curr_act = exp_cfg["camera_actions"][0]
    curr_act_idx = 0

    # Cumulative sum of the frames per prompt and per camera action
    prmpt_frames_cumsum = [sum(exp_cfg["frames_per_prompt"][:i+1]) for i in range(len(exp_cfg["frames_per_prompt"]))]
    act_frames_cumsum = [sum(exp_cfg["frames_per_cam_action"][:i+1]) for i in range(len(exp_cfg["frames_per_cam_action"]))]

    frames = [curr_img]  # Stores the frames for the final gif
    for i in range(prmpt_frames_cumsum[-1]):
        # Checks whether to update the prompt embeddings
        if curr_prmpt_idx+1 < len(prompt_embeds):
            if i == prmpt_frames_cumsum[curr_prmpt_idx]:
                curr_prmpt_idx += 1
                curr_prmpt = prompt_embeds[curr_prmpt_idx]

        # Checks whether to update the camera action
        if curr_act_idx+1 < len(exp_cfg["camera_actions"]):
            if i == act_frames_cumsum[curr_act_idx]:
                curr_act_idx += 1
                curr_act = exp_cfg["camera_actions"][curr_act_idx]

        # Samples random noise
        latent_noise = sd_model.load_noise(
            "",
            exp_cfg["height"],
            exp_cfg["width"],
            1,
            exp_cfg["seed_per_frame"][i] if i < len(exp_cfg["seed_per_frame"]) else i
        )

        margin_height = int(exp_cfg["height"] * exp_cfg["translation_factor"]) // exp_cfg["num_filler_frames"] * exp_cfg["num_filler_frames"]
        margin_width = int(exp_cfg["width"] * exp_cfg["translation_factor"]) // exp_cfg["num_filler_frames"] * exp_cfg["num_filler_frames"]
        mask_img = np.ones((exp_cfg["height"], exp_cfg["width"])) * 255

        prev_img = curr_img  # Used to produce filler frames between the previous and current frame
        if curr_act == "up":
            mask_img[margin_height:, :] = 0
            mask_image = Image.fromarray(np.uint8(mask_img)).convert('RGB')
            in_img = curr_img.transform(
                (exp_cfg["width"], exp_cfg["height"]),
                Image.AFFINE,
                (1, 0, 0, 0, 1, -margin_height),
                resample=Image.BICUBIC
            )
        elif curr_act == "down":
            mask_img[:-margin_height, :] = 0
            mask_image = Image.fromarray(np.uint8(mask_img)).convert('RGB')
            in_img = curr_img.transform(
                (exp_cfg["width"], exp_cfg["height"]),
                Image.AFFINE,
                (1, 0, 0, 0, 1, margin_height),
                resample=Image.BICUBIC
            )
        elif curr_act == "right":
            mask_img[:, :-margin_width] = 0
            mask_image = Image.fromarray(np.uint8(mask_img)).convert('RGB')
            in_img = curr_img.transform(
                (exp_cfg["width"], exp_cfg["height"]),
                Image.AFFINE,
                (1, 0, margin_width, 0, 1, 0),
                resample=Image.BICUBIC
            )
        elif curr_act == "left":
            mask_img[:, margin_width:] = 0
            mask_image = Image.fromarray(np.uint8(mask_img)).convert('RGB')
            in_img = curr_img.transform(
                (exp_cfg["width"], exp_cfg["height"]),
                Image.AFFINE,
                (1, 0, -margin_width, 0, 1, 0),
                resample=Image.BICUBIC
            )
        elif curr_act == "backwards":
            mask_img[margin_height//2:-margin_height//2, margin_width//2:-margin_width//2] = 0
            mask_image = Image.fromarray(np.uint8(mask_img)).convert('RGB')
            downsized_img = curr_img.resize((exp_cfg["width"] - margin_width, exp_cfg["height"] - margin_height))
            in_img = Image.new(downsized_img.mode, (exp_cfg["width"], exp_cfg["height"]), (0, 0, 0))
            in_img.paste(downsized_img, (margin_width//2, margin_height//2))

        # Runs the inference process for Stable Diffusion
        image_embeds, images = sd_model.run_sd_inference(curr_prmpt, latent_noise, in_img, mask_image)

        if curr_act != "backwards":
            in_img.paste(images[0][-1], mask=mask_image.convert('L'))
            curr_img = in_img
        else:
            curr_img = images[0][-1]

        # Stores the experiment results
        utils.save_sd_results(
            output_path=output_path,
            prompt_emb=curr_prmpt.cpu(),
            latent_noise=latent_noise[0].unsqueeze(0).cpu(),
            image_embed=image_embeds[0][-1].cpu(),
            image=curr_img,
            file_name=f"frame-{i}"
        )

        # Produces filler frames between the previous and the current frame 
        for fill_frame_idx in range(1, exp_cfg["num_filler_frames"]):
            add_h = margin_height // exp_cfg["num_filler_frames"] * fill_frame_idx
            add_w = margin_width // exp_cfg["num_filler_frames"] * fill_frame_idx
            filler_frame = Image.new("RGB", (exp_cfg["width"], exp_cfg["height"]), (0, 0, 0))

            if curr_act == "up":
                prev_frame = prev_img.crop((0, 0, exp_cfg["width"], exp_cfg["height"] - add_h))
                curr_frame = curr_img.crop((0, margin_height - add_h, exp_cfg["width"], margin_height))
                filler_frame.paste(curr_frame, (0, 0))
                filler_frame.paste(prev_frame, (0, curr_frame.height))
            elif curr_act == "down":
                prev_frame = prev_img.crop((0, add_h, exp_cfg["width"], exp_cfg["height"]))
                curr_frame = curr_img.crop((0, exp_cfg["height"] - margin_height, exp_cfg["width"], exp_cfg["height"] - margin_height + add_h))
                filler_frame.paste(prev_frame, (0, 0))
                filler_frame.paste(curr_frame, (0, prev_frame.height))
            elif curr_act == "right":
                prev_frame = prev_img.crop((add_w, 0, exp_cfg["width"], exp_cfg["height"]))
                curr_frame = curr_img.crop((exp_cfg["width"] - margin_width, 0, exp_cfg["width"] - margin_width + add_w, exp_cfg["height"]))
                filler_frame.paste(prev_frame, (0, 0))
                filler_frame.paste(curr_frame, (prev_frame.width, 0))
            elif curr_act == "left":
                prev_frame = prev_img.crop((0, 0, exp_cfg["width"]-add_w, exp_cfg["height"]))
                curr_frame = curr_img.crop((margin_width-add_w, 0, margin_width, exp_cfg["height"]))
                filler_frame.paste(curr_frame, (0, 0))
                filler_frame.paste(prev_frame, (curr_frame.width, 0))
            elif curr_act == "backwards":
                filler_frame = curr_img.crop(
                    (margin_width//2-add_w//2,
                    margin_height//2-add_h//2,
                    exp_cfg["width"]-margin_width//2+add_w//2,
                    exp_cfg["height"]-margin_height//2+add_h//2)
                )
                filler_frame = filler_frame.resize((exp_cfg["width"], exp_cfg["height"]))
            else:
                continue
            filler_frame.save(f"{output_path}/images/{i}_filler_frame_{fill_frame_idx}.png")
            frames.append(filler_frame)
        frames.append(curr_img)

    # Produces a gif to visualize the outpaint walk
    utils.produce_gif([frames], output_path, exp_cfg["gif_frame_dur"])
    print("Experiment finished")