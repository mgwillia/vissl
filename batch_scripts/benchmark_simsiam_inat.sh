#!/bin/bash

#SBATCH --job-name=b_simsiam_inat
#SBATCH --output=outfiles/b_simsiam_inat.out.%j
#SBATCH --error=outfiles/b_simsiam_inat.out.%j
#SBATCH --time=72:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

if [ ! -d /scratch0/mgwillia/inat_comp_2021/val ]; then
    srun bash -c "echo 'inat21 not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -P -p 16 /fs/vulcan-datasets/inat_comp_2021 /scratch0/mgwillia/"
fi

srun bash -c 'hostname; python ./tools/run_distributed_engines.py \
                config=benchmark/linear_image_classification/inaturalist21/eval_resnet_8gpu_transfer_inaturalist21_linear \
                config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/simsiam_r50_100.torch \
                config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=network \
                config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                config.DATA.TRAIN.DATA_PATHS=["/scratch0/mgwillia/inat_comp_2021/train"] \
                config.DATA.TEST.DATA_PATHS=["/scratch0/mgwillia/inat_comp_2021/val"] \
                config.CHECKPOINT.DIR="/vulcanscratch/mgwillia/vissl/checkpoints_bench_simsiam_inat" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=32'
