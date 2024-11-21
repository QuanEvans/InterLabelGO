# InterLabelGO+

InterLabelGO+ combines a deep learning model (InterLabelGO) and sequence homology search to perform predict a query protein's biological function, in the form of Gene Ontology (GO) terms. InterLabelGO uses the last three layers the [ESM2](https://github.com/facebookresearch/esm) large language model to extract sequence features, which are then learned by a series of neural networks to predict GO terms under a new loss function that incorporates label imbalances and inter-label dependencies. These deep learning predicted terms are then combined with [DIAMOND](https://github.com/bbuchfink/diamond) search results through a dynamic weighting scheme to derive the consensus prediction. InterLabelGO+ (as team [Evans](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/466971)) was ranked among the top teams in the recent CAFA5 challenge. 

## System Requirements

InterLabelGO is developed under a Linux environment with the following software:

- Python 3.11.5
- CUDA 12.1
- cuDNN 8.9.6
- DIAMOND v2.1.8
- NVIDIA drivers v.535.129.03

Python packages are detailed separately in `requirements.txt`.

## Set up InterLabelGO+

1. Downloading the xz file from [https://seq2fun.dcmb.med.umich.edu/InterLabelGO/InterLabelGO+.tar.xz](https://seq2fun.dcmb.med.umich.edu/InterLabelGO/InterLabelGO+.tar.xz)
2. Extract the file
3. Run `setup_env.sh` to create an InterLabelGO conda environment and download ESM2 models

Alternatively, you can create the conda environment manually:

```bash
conda create -n InterLabelGO python=3.11.5 -y
conda activate InterLabelGO
pip install -r requirements.txt
```

Then download ESM2 models:

```bash
this_file_path=$(dirname $(readlink -f $0))
mode_pt_url="https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
regression_url="https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt"
save_path=$this_file_path/Data/esm_models
mkdir -p $save_path
wget -P $save_path $mode_pt_url
wget -P $save_path $regression_url
```

This will download the ESM2 models into `Data/esm_models`.

To update the current GOA database, run:

```bash
python update_data.py
```

This will download the latest GOA database and convert it into the required format.

## Data Processing

Run the following command:

```bash
python prepare_data.py --make_db --ia
```

This will:
- Convert raw data (`train_terms.tsv` and `train_seq.fasta`) into required training data
- Create a DIAMOND database for the training data (`--make_db`)
- Create an Information Content file for the training data (`--ia`)
- Extract the ESM embeddings for the training data

To use stratified multi-label in k-fold (time-consuming), add the `--stratifi` argument.

All paths are specified in `settings.py`.

## Model Usage

### 1. Prediction

```bash
python predict.py -w example -f example/seq.fasta --use_gpu
```

This will predict GO terms for the example sequence.

Additional options:
- Add `--seqid` to use sequence ID for combination of alignments and neural network method
- Add `-c` to continue from the last unfinished prediction (won't check for changes in the fasta file)

### 2. Retrain Models

```bash
python train_InterLabelGO.py
```

Training configuration is specified in `settings.py`.

## Notes

1. The data processing code will overwrite the original data, and the training code will overwrite the original model.

## Citation

Quancheng Liu, Chengxin Zhang, Lydia Freddolino (2024)
[InterLabelGO+: unraveling label correlations in protein function prediction](https://doi.org/10.1093/bioinformatics/btae655)
Bioinformatics, 40(11): btae655.

