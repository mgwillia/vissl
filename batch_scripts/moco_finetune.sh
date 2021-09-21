#!/bin/bash

#SBATCH --job-name=moco_f_inet
#SBATCH --output=outfiles/moco_f_inet.out.%j
#SBATCH --error=outfiles/moco_f_inet.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:gtx1080ti:8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

module load cuda/10.1.243

#srun bash -c "mkdir -p /scratch0/mgwillia"
#srun bash -c "rsync -r /fs/vulcan-datasets/imagenet /scratch0/mgwillia/"

srun bash -c 'hostname; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./tools/run_distributed_engines.py config=./pretrain/moco/moco_finetune_1node_resnet \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=1 \
    config.DATA.TRAIN.DATA_PATHS=["/fs/vulcan-datasets/imagenet"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/vulcanscratch/mgwillia/unsupervised-classification/backbones/moco_r50_200.torch" \
    config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_moco_f" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=32'
