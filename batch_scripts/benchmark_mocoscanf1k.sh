#!/bin/bash

#SBATCH --job-name=b_mcscnf_1k
#SBATCH --output=outfiles/b_mcscnf_1k.out.%j
#SBATCH --error=outfiles/b_mcscnf_1k.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

srun bash -c 'hostname; python ./tools/run_distributed_engines.py \
                config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/mocoscanf1k_r50_1.torch \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/imagenet/train"] \
                config.DATA.TEST.DATA_PATHS=["/scratch0/mgwillia/imagenet/val"] \
                config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_bench_mocoscan_1k" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64 \
                config.OPTIMIZER.num_epochs=1'
