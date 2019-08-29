# README

This project contains the source code of ADT 1.0, a Lua-based adversarial discriminative sim-to-real transfer architecture, necessary to reproduce the experiments described in the paper:

> Fangyi Zhang, Jürgen Leitner, Zongyuan Ge, Michael Milford, Peter Corke, "AdversarialDiscriminative Sim-to-real Transfer of Visuo-motor Policies," International Journal of RoboticsResearch (IJRR), 2019. (https://doi.org/10.1177%2F0278364919870227)

*If you use the code, datasets or models for your academic research, please cite the paper.*


### Installation instructions

The installation requires Linux with apt-get.

Note: In order to run the GPU version of DQN, you should additionally have the
NVIDIA CUDA (version 5.5 or later) toolkit installed prior to the Torch
installation below.
This can be downloaded from https://developer.nvidia.com/cuda-toolkit
and installation instructions can be found at
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux


To train modular networks for 7 DoF reaching, the following components must be installed:
* LuaJIT and Torch 7.0
* nngraph

To install all of the above in a subdirectory called 'torch', it should be enough to run

    ./install_dependencies.sh

from the base directory of the package.

Note: The above install script will install the following packages via apt-get:
build-essential, gcc, g++, cmake, curl, libreadline-dev, git-core, libjpeg-dev,
libpng-dev, ncurses-dev, imagemagick, unzip


### List of scripts
* 7DoF_reaching_ADT_adapt_p                A script to adapt a perception module for real scenarios using ADT
* 7DoF_reaching_ADT_finetune_e2e           A script to end-to-end fine-tune a combined network (perception + control) for better hand-eye coordination using ADT
* 7DoF_reaching_supervised_c               A script to train a control module using supervised learning
* 7DoF_reaching_supervised_finetune_e2e    A script to end-to-end fine-tune a combined network (perception + control) using supervised learning for the comparison with the ADT one
* 7DoF_reaching_supervised_p               A script to train a perception module using supervised learning
* 7DoF_reaching_test_p_offline             A script to test perception modules using the real testset
* 7DoF_reaching_test_c                     A script to test control module
* 7DoF_reaching_test_e2e                   A script to test a combined network (perception + control) with raw-pixel images as inputs


### Training and testing

Prior to training or testing, you need to:
- download required [datasets](https://drive.google.com/drive/folders/1peyIP4kLna8OhZqCubCR-hDzAJZKi0DP?usp=sharing) and [models](https://drive.google.com/file/d/1fbj6JbtqGIyRymdy19NtWQn0O8dQwTk8/view?usp=sharing) and place them in the 'deep_manipulation' folder;
- change the settings in relevant scripts (e.g., 7DoF_reaching_ADT_adapt_p in the root) and simulator enties (e.g., vrep_baxter_picking_inhand_cam_dataset_adt_adapt_p.lua in the folder "simulator").

Now, it should then be sufficient to run various scripts, e.g.,

    ./7DoF_reaching_ADT_adapt_p <experiment name>

Note: You would have to change the gpu settings to use cpu or gpu. On a system with more than one GPU, the training can be launched on a
specified GPU, e.g. by

    gpu=0

If use CPU, set it to -1, otherwise, the program will use GPU.


### Options

Options to ADT are set within different scripts. You may,
for example, want to change the frequency at which information is output
to stdout by setting 'prog_freq' to a different value. Please set accordingly in the script, comments are provided to introduce the functionality of each parameter.
