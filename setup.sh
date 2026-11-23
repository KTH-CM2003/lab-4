#!/bin/bash
# Sets up everything needed for Lab 4:
#   1. the conda environment "cm2003-lab4" (see environment.yml)
#   2. a Jupyter kernel with the same name
# The hippocampus dataset is downloaded by the notebooks themselves (download=True).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_NAME=cm2003-lab4

source "$(conda info --base)/etc/profile.d/conda.sh"

# Python comes from conda-forge only (--override-channels), so Anaconda's "defaults" channel and
# its Terms of Service prompt are not used; the lab packages come from requirements.txt via pip.
if conda env list | grep -qE "^${ENV_NAME} |/${ENV_NAME}$"; then
  echo "Environment ${ENV_NAME} already exists, updating its packages"
else
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.13 pip
fi

conda activate "$ENV_NAME"
python -m pip install -r requirements.txt
python -m ipykernel install --user --name="$ENV_NAME" --display-name "CM2003 Lab 4"

echo
echo "Setup complete! In Jupyter, select the kernel \"CM2003 Lab 4\" (Kernel > Change kernel)."
