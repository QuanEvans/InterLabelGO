#!/bin/bash

conda_path=$(conda info --base)
source "$conda_path/etc/profile.d/conda.sh"

this_file_path=$(dirname $(readlink -f $0))
# create conda env
conda create -n InterLabelGO python=3.11.5 -y
conda activate InterLabelGO
pip install -r $this_file_path/requirements.txt

# # download esm2_3b_model
mode_pt_url="https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
regression_url="https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt"
save_path=$this_file_path/Data/esm_models
mkdir -p $save_path

wget -P $save_path $mode_pt_url
wget -P $save_path $regression_url
