#!/bin/bash

#SBATCH --job-name=swav_f_inet                                # sets the job name
#SBATCH --output=outfiles/swav_f_inet.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/swav_f_inet.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/10.1.243                                  # run any commands necessary to setup your environment

srun bash -c 'hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/run_distributed_engines.py config=./pretrain/swav/swav_distill_8node_resnet \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
    config.DATA.TRAIN.DATA_PATHS=["/fs/vulcan-datasets/imagenet"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="swav_pretrained.torch" \
    config.CHECKPOINT.DIR="./checkpoints_swav" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=100'
