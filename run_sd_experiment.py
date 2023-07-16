import argparse
import os

import yaml

import experiments
import models


def load_model(model_cfg):
    """
    Loads a Stable Diffusion model instance from the diffusers library.

    :param model_cfg: Dictionary containing the model configurations.
    :return: Stable Diffusion model instance.
    """
    if model_cfg["model_identifier"] == "txt2img":
        sd_model = models.Txt2Img(model_cfg)
    elif model_cfg["model_identifier"] == "img2img":
        sd_model = models.Img2Img(model_cfg)
    elif model_cfg["model_identifier"] == "inpaint":
        sd_model = models.Inpaint(model_cfg)
    return sd_model


def run_sd_experiment(cfg_path, exp_cfg, sd_model):
    """
    Runs a Stable Diffusion experiment.

    :param cfg_path: String path to the loaded configuration file.
    :param exp_cfg: Dictionary containing the experiment configurations.
    :param sd_model: Stable Diffusion model instance.
    """
    print(f"Starting {exp_cfg['exp_identifier']} experiment")
    if exp_cfg["exp_identifier"] == "single-inference":
        experiments.run_single_inference(cfg_path, exp_cfg, sd_model)
    elif exp_cfg["exp_identifier"] == "visualize-diffusion":
        experiments.run_visualize_diffusion(cfg_path, exp_cfg, sd_model)
    elif exp_cfg["exp_identifier"] == "random-walk":
        experiments.run_random_walk(cfg_path, exp_cfg, sd_model)
    elif exp_cfg["exp_identifier"] == "interpolation":
        experiments.run_interpolation(cfg_path, exp_cfg, sd_model)
    elif exp_cfg["exp_identifier"] == "diffevolution":
        experiments.run_diffevolution(cfg_path, exp_cfg, sd_model)
    elif exp_cfg["exp_identifier"] == "outpaint-walk":
        experiments.run_outpaint_walk(cfg_path, exp_cfg, sd_model)


def main():
    parser = argparse.ArgumentParser(description="Run a Stable Diffusion experiment by licking to a configuration file.")
    parser.add_argument('--exp_config', type=str, default="./configs/experiments/txt2img/single_inference.yaml",
                        help='Path to the experiment configuration file.')
    args = parser.parse_args()

    assert os.path.isfile(args.exp_config)

    with open(args.exp_config) as yaml_file:
        exp_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    sd_model = load_model(exp_cfg)
    run_sd_experiment(args.exp_config, exp_cfg, sd_model)


if __name__ == "__main__":
    main()