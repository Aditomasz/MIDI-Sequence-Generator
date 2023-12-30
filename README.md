# MIDI Creation AI

A Gradio web UI for using GPT structure for creating music in .mid files.

## Features

* Data Preparation
* Model Creation
* Model Training
* Generating .mid files

## Documentation

To learn how to use the various features, check out the Documentation: 
https://github.com/Aditomasz/MIDI-Sequence-Generator/wiki
// currently not fully implemented

## Requirements

This program requires cuda to be installed alongside machine with cuda compatible GPU.
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

## Instalation

### One-click installers

#### On Windows

1) Clone the repository
2) Run "run.bat"
3) Have fun!

#### On Linux

One-click installer is not currently supported on Linux, proceed to manual installation.
// Bash script is planned for Future Release

##### How it works

The script creates a folder called `installer_files` where it sets up a Conda environment using Miniconda. The installation is self-contained: if you want to reinstall, just delete `installer_files` and run the start script again.
To launch the webui in the future after it is already installed, run the same `start` script.

### Manual Installation using Conda

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands ([source](https://educe-ubc.github.io/conda.html)):
```bash
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

#### 1. Create a new conda environment

```
conda create -n myenv python=3.11
conda activate myenv
```
Replace 'myenv' with the name you want to give to your environment.

#### 2. Install the web UI

```
git clone https://github.com/Aditomasz/MIDI-Sequence-Generator
cd MIDI-Sequence-Generator
pip install -r requirements.txt
```

## Acknowledgment

This repository is using slightly modified version of [Midi Neural Processor by jason9693](https://github.com/jason9693/midi-neural-processor).
