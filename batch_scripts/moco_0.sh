#!/bin/bash

#SBATCH --job-name=moco_0
#SBATCH --output=outfiles/moco_0.out.%j
#SBATCH --error=outfiles/moco_0.out.%j
#SBATCH --time=72:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

if [ ! -d /scratch0/mgwillia/imagenet/val ]; then
    srun bash -c "echo 'imagenet not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 8 /fs/vulcan-datasets/imagenet /scratch0/mgwillia/"
fi

srun bash -c 'hostname; python ./tools/run_distributed_engines.py config=./pretrain/moco/moco_1node_resnet \
    config.SEED_VALUE=0 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
    config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/imagenet"] \
    config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_moco_0" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64'
