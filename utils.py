from datetime import datetime
import os
import shutil

import numpy as np
import torch


def generate_output_folder(model_identifier, exp_identifier, cfg_path, dest_path, embeddings=True, images=True, gifs=True):
    """
    Generates an output folder for storing the experiment configuration and the generated outputs. The folder name
    consists of the current date, time, model identifier and experiment identifier.

    :param model_identifier: String containing the model identifier from the configuration file.
    :param exp_identifier: String containing the experiment identifier from the configuration file.
    :param cfg_path: The string path to the configuration file, which was loaded for the experiment.
    :param dest_path: The string path where the output folder should be placed.
    :param embeddings: Whether to generate an embeddings folder.
    :param images: Whether to generate an images folder.
    :param gifs: Whether to generate a gifs folder.
    :return: The string path to the newly generated output folder.
    """
    folder_name = f"{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}_{model_identifier}_{exp_identifier}"
    folder_path = f"{dest_path}/{folder_name}"

    # Generates the subfolders
    os.makedirs(f"{folder_path}/configs")
    os.makedirs(f"{folder_path}/embeddings") if embeddings else None
    os.makedirs(f"{folder_path}/images")  if images else None
    os.makedirs(f"{folder_path}/gifs")  if gifs else None

    # Copies the experiment configuration
    shutil.copy(cfg_path, f"{folder_path}/configs/{cfg_path.split('/')[-1]}")

    print(f"Output folder generated at {folder_path}\n")
    return folder_path


def save_sd_results(output_path, prompt_emb, latent_noise, image_embed, image, file_name):
    """
    Saves the experiment results produced by Stable Diffusion.

    :param output_path: String path to the experiment folder for storing the experiment results.
    :param prompt_emb: Torch tensor of the encoded prompt embedding, which served as input for Stable Diffusion.
    :param latent_noise: Torch tensor of the latent gaussian noise tensor, which served as input for Stable Diffusion.
    :param image_embed: Torch tensor of the image embedding produced by Stable Diffusion.
    :param image: PIL Image of the decoded image embedding, produced by Stable Diffusion.
    :param file_name: String name of the file, for storing the results.
    """
    torch.save(
        {
            "prompt_embed": prompt_emb,
            "latent_noise": latent_noise,
            "image_embed": image_embed
        },
        f"{output_path}/embeddings/{file_name}.pt"
    )
    image.save(f"{output_path}/images/{file_name}.png")


def produce_gif(images, output_path, gif_frame_dur):
    """
    Produces a gif from individual frames.

    :param images: A list of lists containing individual frames. Each list represents a different gif.
    :param output_path: The string path to the experiment results folder.
    :param gif_frame_dur: Specifies the frame duration in milliseconds for the produced gifs.
    """
    for i in range(len(images)):
        images[i][0].save(f"{output_path}/gifs/output-{i}.gif", format="GIF",
                           append_images=images[i][1:], save_all=True, duration=gif_frame_dur, loop=0)


def interpolate(x, y, steps, interpolation_method="slerp"):
    """
    Generates interpolations of two vectors/tensors.

    :param x: First torch vector/tensor to interpolate.
    :param y: Second torch vector/tensor to interpolate
    :param steps: The amount of interpolation steps
    :param interpolation_method: Specifies the interpolation method (can be either "lerp" for linear interpolation or
    "slerp" for spherical linear interpolation).
    :return: A list of interpolated vectors/tensors where the first and last element correspond to x and y respectively.
    """
    interpolations = []

    if x is None or y is None:
        # Baseline case
        return [None for _ in range(steps + 2)]

    for t in np.linspace(0, 1, steps + 2):
        if interpolation_method == "slerp":
            # Spherical linear interpolation
            interpolations.append(slerp(x, y, t))
        else:
            # Linear interpolation
            interpolations.append(torch.lerp(x, y, t))
    return interpolations


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Generates a weighted spherical linear interpolation between two vectors/tenors.
    """
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().detach().numpy()
        v1 = v1.cpu().detach().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)
    return v2