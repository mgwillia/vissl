#!/bin/bash

#SBATCH --job-name=b_cub
#SBATCH --output=outfiles/b_cub.out.%j
#SBATCH --error=outfiles/b_cub.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

INDICES=(0 1 2 3 4 5 6 7 8 9 10)
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "supervised_r50" "swav_r50_800" "simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_1000")
CHECKPOINTS_DIRS=("chkpts_b_cub_btwins" "chkpts_b_cub_dc" "chkpts_b_cub_moco" "chkpts_b_cub_sclr" "chkpts_b_cub_simsiam" "chkpts_b_cub_sup" "chkpts_b_cub_swav" "chkpts_b_cub_sclr100" "chkpts_b_cub_sclr200" "chkpts_b_cub_sclr400" "chkpts_b_cub_sclr1000")
STATE_DICT_KEY_NAMES=("classy_state_dict" "network" "network" "classy_state_dict" "network" "network" "classy_state_dict" "classy_state_dict" "classy_state_dict" "classy_state_dict" "classy_state_dict")

if [ ! -d /scratch0/mgwillia/CUB_200_2011 ]; then
    srun bash -c "echo 'cub not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "rsync -r /vulcanscratch/mgwillia/CUB_200_2011 /scratch0/mgwillia/"
fi

for i in ${INDICES[@]}; do
    srun bash -c "hostname; python ./tools/run_distributed_engines.py \
                    config=benchmark/linear_image_classification/cub/eval_resnet_8gpu_transfer_cub_linear \
                    config.MODEL.TRUNK.RESNETS.DEPTH=50 \
                    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/${BACKBONES[$i]}.torch \
                    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=${STATE_DICT_KEY_NAMES[$i]} \
                    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                    config.DATA.TRAIN.DATA_PATHS=['/scratch0/mgwillia/CUB_200_2011/train'] \
                    config.DATA.TEST.DATA_PATHS=['/scratch0/mgwillia/CUB_200_2011/val'] \
                    config.CHECKPOINT.DIR=/vulcanscratch/mgwillia/vissl/${CHECKPOINTS_DIRS[$i]} \
                    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64"
done
