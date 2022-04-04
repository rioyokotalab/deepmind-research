#!/bin/bash
#YBATCH -r a6000_4
#SBATCH -N 1
#SBATCH -J pretr_byol

START_TIMESTAMP=$(date '+%s')

# ======== Variables ========

job_id_base=$SLURM_JOBID

git_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)

data_root="/mnt/nfs/datasets/waymo_opendata_root/ILSVRC2012"
imagenet_name="imagenet2012"

local_ssd_path="$HINADORI_LOCAL_SCRATCH"

epochs=1000
date_str=$(date '+%Y%m%d_%H%M%S')
checkpoint_root="$git_root/byol/tmp/pretrain/byol_checkpoints"
mkdir -p "$checkpoint_root"

# ======== Copy ========

COPY_START_TIMESTAMP=$(date '+%s')

local_data_root="$local_ssd_path/ILSVRC2012"
mkdir -p "$local_data_root"

rsync -avz "$data_root/$imagenet_name" "$local_data_root"
COPY_END_TIMESTAMP=$(date '+%s')

COPY_E_TIME=$(($COPY_END_TIMESTAMP-$COPY_START_TIMESTAMP))
echo "copy time: $COPY_E_TIME s"

export TFDS_DATA_DIR="$local_data_root"

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

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"
