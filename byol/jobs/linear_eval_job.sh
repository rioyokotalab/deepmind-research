#!/bin/bash
#YBATCH -r am_4 
#SBATCH -N 1 
#SBATCH -J eval_byol

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

# date_str=$(date '+%Y%m%d_%H%M%S')
# out_put="$git_root/tmp/byol_checkpoints/$date_str"
out_put="$git_root/tmp/byol_checkpoints/test"
mkdir -p "$out_put" 

# python main_loop.py \

pushd "$git_root"

set -x

python -m byol.main_loop \
    --experiment_mode="linear-eval" \
    --worker_mode="train" \
    --checkpoint_root="$out_put"

set +x

popd

    # --worker_mode="eval" \
    # --pretrain_epochs=40 \
    # --batch_size=2048 \
