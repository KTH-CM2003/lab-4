# CM2003 Lab 4: Segmentation, U-Net and MONAI

## Introduction
This lab is designed to introduce you to the concepts of medical image segmentation and the U-Net architecture. You will also get hands-on experience with training a U-Net model using PyTorch and [MONAI](https://monai.io/), a PyTorch-based framework for medical imaging.

You will segment the hippocampus in brain MRI, using Task04 of the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). The notebooks download the dataset (about 30 MB) into `cm2003/datasets/` the first time you run them.

## Setting up the environment

Pick **one** of the options below.

### Option 1: On MAIA

Your MAIA workspace already has a ready-made environment called **`DL_labs_GPU`** with PyTorch and MONAI, so there is
nothing to install. Clone your repository in a JupyterLab terminal (*File > New > Terminal*), open a
notebook and select the kernel **DL_labs_GPU** (*Kernel > Change Kernel...*). If it is not in the
list, refresh the browser tab.

### Option 2: On your own computer with conda

Install [Miniforge](https://github.com/conda-forge/miniforge) (or use an existing
Anaconda/Miniconda), go to the repository folder in a terminal and run `./setup.sh` (macOS / Linux).
On Windows, run the same steps by hand in the *Miniforge Prompt*:
```bash
conda env create -f environment.yml
conda activate cm2003-lab4
python -m ipykernel install --user --name=cm2003-lab4 --display-name "CM2003 Lab 4"
```
Then open a notebook and select the kernel **CM2003 Lab 4**.

### Option 3: On your own computer with uv

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
uv run python -m ipykernel install --user --name=cm2003-lab4 --display-name "CM2003 Lab 4"
```

### GPU or CPU?

Training the U-Nets is much faster on a GPU, so we recommend MAIA (or another computer with an
NVIDIA GPU) for this lab. The models need at most about 2 GB of GPU memory. On a laptop, training
the U-Net in Part 1 takes several minutes per epoch and needs about 3 GB of free memory, so close
other notebooks (*Kernel > Shut Down Kernel*) before you start. The notebooks print the device they use: `cuda` (NVIDIA GPU), `mps`
(Apple-silicon Mac) or `cpu`.

## Lab structure

The lab is divided into three parts. Do them in this order, since the second part uses the model
trained in the first:

1. **Part 1: Introduction to Segmentation and U-Net:** `Segmentation and UNET.ipynb`. You will
   implement the U-Net in `cm2003/unet.py` and the Dice score in `cm2003/metrics.py`. The trained
   model is saved as `trained_unet.pth`.
2. **Part 2: Visualizing attention maps:** `Attention maps.ipynb`. You will use Grad-CAM to see
   which regions the U-Net from Part 1 relies on.
3. **Part 3: Training a U-Net model with MONAI:** `MONAI.ipynb`. You will first train a 2D U-Net
   with MONAI, and then a 3D U-Net the way it is usually done in practice: on patches at native
   resolution, with data augmentation, sliding window inference and evaluation per volume.

## Submission

Please submit your code and a short overview report of the results from the lab on GitHub.
