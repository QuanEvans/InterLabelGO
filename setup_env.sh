#!/bin/bash

# Define the base directory for Conda installation
basedir=$(dirname $(readlink -f $0))
conda_dir="$basedir/conda"

# Check if Conda is installed; if not, download and install it
if [ ! -d "$conda_dir" ]; then
  echo "Conda not found. Installing Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $basedir/miniconda.sh
  bash $basedir/miniconda.sh -b -p $conda_dir
  rm $basedir/miniconda.sh
fi

# Initialize Conda
conda_path="$conda_dir"
source "$conda_path/etc/profile.d/conda.sh"

this_file_path=$(dirname $(readlink -f $0))

# Create Conda environment
conda create -n InterLabelGO python=3.11.5 -y
conda activate InterLabelGO
pip install -r $this_file_path/requirements.txt

# Download esm2_3b_model if not already downloaded
mode_pt_url="https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
regression_url="https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt"
save_path=$this_file_path/Data/esm_models
mkdir -p $save_path

if [ ! -f "$save_path/$(basename $mode_pt_url)" ]; then
  echo "Downloading ESM model..."
  wget -P $save_path $mode_pt_url
else
  echo "ESM model already exists, skipping download."
fi

if [ ! -f "$save_path/$(basename $regression_url)" ]; then
  echo "Downloading regression model..."
  wget -P $save_path $regression_url
else
  echo "Regression model already exists, skipping download."
fi
