#!/bin/bash

#SBATCH --job-name=b_cub_1
#SBATCH --output=outfiles/b_cub_1.out.%j
#SBATCH --error=outfiles/b_cub_1.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

INDICES=(0 1 2 3 4 5 6)
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
CHECKPOINTS_DIRS=("chkpts_b_cub_btwins_1" "chkpts_b_cub_dc_1" "chkpts_b_cub_moco_1" "chkpts_b_cub_sclr_1" "chkpts_b_cub_simsiam_1" "chkpts_b_cub_sup_1" "chkpts_b_cub_swav_1")
STATE_DICT_KEY_NAMES=("classy_state_dict" "network" "network" "classy_state_dict" "network" "network" "classy_state_dict")

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
