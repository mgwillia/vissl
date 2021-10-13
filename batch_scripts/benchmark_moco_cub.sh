#!/bin/bash

#SBATCH --job-name=b_moco_cub
#SBATCH --output=outfiles/b_moco_cub.out.%j
#SBATCH --error=outfiles/b_moco_cub.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

if [ ! -d /scratch0/mgwillia/CUB_200_2011 ]; then
    srun bash -c "echo 'cub not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "rsync -r /vulcanscratch/mgwillia/CUB_200_2011 /scratch0/mgwillia/"
fi

srun bash -c 'hostname; python ./tools/run_distributed_engines.py \
                config=benchmark/linear_image_classification/cub/eval_resnet_8gpu_transfer_cub_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/moco_r50_800.torch \
                config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=network \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/CUB_200_2011/train"] \
                config.DATA.TEST.DATA_PATHS=["/scratch0/mgwillia/CUB_200_2011/val"] \
                config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_bench_moco_cub" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64'
