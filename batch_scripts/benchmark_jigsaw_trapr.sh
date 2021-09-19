#!/bin/bash

#SBATCH --job-name=jigsaw_lin                                # sets the job name
#SBATCH --output=outfiles/jigsaw_trapr_lin.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/jigsaw_trapr_lin.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:gtx1080ti:8
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/10.1.243                                  # run any commands necessary to setup your environment

srun bash -c "mkdir -p /scratch0/mgwillia"
srun bash -c "rsync -r /fs/vulcan-datasets/imagenet /scratch0/mgwillia/"

srun bash -c 'hostname; python ./tools/run_distributed_engines.py \
                config=./benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/vissl/checkpoints_jigsaw_trapr/model_final_checkpoint_phase104.torch \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/imagenet/train"] \
                config.DATA.TEST.DATA_PATHS=["/scratch0/mgwillia/imagenet/val"] \
                config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_bench_jigsaw_trapr"'