#!/bin/bash

#SBATCH --job-name=b_dc
#SBATCH --output=outfiles/b_dc.out.%j
#SBATCH --error=outfiles/b_dc.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

srun bash -c 'hostname; python ./tools/run_distributed_engines.py \
                config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/dcv2_r50_800.torch \
                config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=network \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/imagenet/train"] \
                config.DATA.TEST.DATA_PATHS=["/scratch0/mgwillia/imagenet/val"] \
                config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_bench_dc" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64'
