#!/bin/bash
#YBATCH -r am_4
#SBATCH -N 1
#SBATCH -J pretr_byol

git_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)

export TFDS_DATA_DIR="/mnt/nfs/datasets/waymo_opendata_root/ILSVRC2012"

# ======== Pyenv ========

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
which python

# ======== Modules ========

source /etc/profile.d/modules.sh

module load cuda/11.2
module load cudnn/cuda-11.2/8.1

# ======== Scripts ========

checkpoint_root="$git_root/byol/tmp/pretrain/byol_checkpoints"
mkdir -p "$checkpoint_root" 

epochs=1000

pushd "$git_root"


python -m byol.main_loop \
    --experiment_mode="pretrain" \
    --worker_mode="train" \
    --checkpoint_root="$checkpoint_root" \
    --pretrain_epochs=$epochs \
    --wandb_runname "pretrain_byol_$epochs" \
    --wandb_project "byol_results"
    # --batch_size=2048 \

popd
