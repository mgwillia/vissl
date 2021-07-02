#!/bin/bash

#SBATCH --job-name=sclr_c10                                # sets the job name
#SBATCH --output=outfiles/sclr_c10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/sclr_c10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load cuda/10.1.243                                  # run any commands necessary to setup your environment

srun bash -c 'hostname; CUDA_VISIBLE_DEVICES=0,1 python ./tools/run_distributed_engines.py config=./pretrain/simclr/simclr_8node_resnet_cifar \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 config.DISTRIBUTED.NUM_NODES=1 \
    config.CHECKPOINT.DIR="./checkpoints_sclr_c10" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256'
