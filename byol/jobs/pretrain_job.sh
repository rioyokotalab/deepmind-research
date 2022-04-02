#!/bin/bash
#YBATCH -r am_8 
#SBATCH -N 1 
#SBATCH -J pretr_byol
#SBATCH --output=slurm%j_batch4096.out

git_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)

pushd "$git_root"

mkdir -p tmp/byol_checkpoints

python -m byol.main_loop \
    --experiment_mode='pretrain' \
    --worker_mode='train' \
    --checkpoint_root='tmp/byol_checkpoints' \
    --pretrain_epochs=40 \
    # --batch_size=2048 \

popd
