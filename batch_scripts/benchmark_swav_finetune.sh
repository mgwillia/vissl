#!/bin/bash

#SBATCH --job-name=b_swav_f_inet                                # sets the job name
#SBATCH --output=outfiles/b_swav_f_inet.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/b_swav_f_inet.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243                                  # run any commands necessary to setup your environment

srun bash -c 'hostname; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./tools/run_distributed_engines.py \
                config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk.base_model._feature_blocks. \
                config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX=module. \
                config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=./checkpoints_swav/checkpoint.tar \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/fs/vulcan-datasets/imagenet"] \
                config.DATA.TEST.DATA_PATHS=["/fs/vulcan-datasets/imagenet"] \
                config.CHECKPOINT.DIR="./checkpoints_bench_swav" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=32'
