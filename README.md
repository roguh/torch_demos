# Hugo's Torch Demos

## 1. Setup NVIDIA

Highly dependent on your OS.

## 2. Install CUDA and AI libraries

These are instructions for Linux.

### Option 1: Pacman

1. cuda
  pacman -S cuda cuda-tools opencl-nvidia

2. distributed
  pacman -S nccl

3. AI libaries
  pacman -S python-pytorch-cuda

can copy files locally between computers to speed up installation

### Option 2: Anaconda

1. Install miniconda (recommended by fastai) or anaconda
  https://docs.anaconda.com/anaconda/install/linux/
  https://conda.io/projects/conda/en/latest/user-guide/install/index.html
  https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

1. Load conda in your shell with `conda init` or `conda init fish`

1. Install mamba for faster and more reliable downloads
   https://github.com/mamba-org/mamba

   Use `mamba` instead of `conda` in the next steps

1. Install fastai
   https://docs.fast.ai/

   `conda install -c fastchan fastai`

1. Install pytorch
   https://pytorch.org/get-started/locally/

   something like: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

   Actually, there's a bug if you try use `mamba install`

   Use: `mamba create -n torch pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`
   to create an environment named `torch`

To manually activate an environment whenever conda is loaded by your shell:
`conda config --set auto_activate_base false`
This helps you keep using your system's Python
