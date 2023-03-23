# Stable-Diffusion-Latent-Space-Explorer (Winter 2022/2023)

## General Information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Advisors:**
* [Prof. Dr. Visvanathan Ramesh](http://www.ccc.cs.uni-frankfurt.de/people/), email: V.Ramesh@em.uni-frankfurt.de

**Institutions:**
  * **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
  * **[AISEL - AI Systems Engineering Lab](http://www.ccc.cs.uni-frankfurt.de/)**

**Project team (A-Z):**
* Alen Smajic

## Setup
### Installation
1. Clone this repository.

       git clone https://github.com/alen-smajic/Stable-Diffusion-Latent-Space-Explorer
       
2. Create a virtual environment.

       python -m venv venv
       
2. Install PyTorch with CUDA ([follow this PyTorch installation](https://pytorch.org/get-started/locally/)).
       
3. Install [diffusers](https://huggingface.co/docs/diffusers/index) and [transformers](https://huggingface.co/docs/transformers/index) librariers.

       pip install diffusers["torch"], transformers
       
4. Optional: Install [xFormers](https://github.com/facebookresearch/xformers) for efficient attention.

       pip install xformers
       
### Model Weights
You can download the model weights using [git-lfs](https://git-lfs.com/).

        git lfs install
        git clone https://huggingface.co/stabilityai/stable-diffusion-2-1
        
The comman above will create a local folder called ./stable-diffusion-2-1 on your disk.

Tested on: 
- [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [Stable Diffusion 2.1 base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- [Stable Diffusion 2 Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

## Instructions

In order to run an experiment, you first need to define the experiment parameters via the config files.
