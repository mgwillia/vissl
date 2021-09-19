#!/bin/bash

#SBATCH --job-name=b_mocoscanv1_128
#SBATCH --output=outfiles/b_mocoscanv1_128.out.%j
#SBATCH --error=outfiles/b_mocoscanv1_128.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

srun bash -c 'hostname; python ./tools/run_distributed_engines.py \
                config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/mocoscanv1f128_r50_20.pth.tar \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/imagenet/train"] \
                config.DATA.TEST.DATA_PATHS=["/scratch0/mgwillia/imagenet/val"] \
                config.CHECKPOINT.DIR="/cfarhomes/mgwillia/vissl/checkpoints_bench_mocoscan_v1_128" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64'
