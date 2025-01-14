#!/bin/bash

#SBATCH --job-name=knn_inet
#SBATCH --output=outfiles/knn_inet.out.%j
#SBATCH --error=outfiles/knn_inet.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.1.243

if [ ! -d /scratch0/mgwillia/imagenet/val ]; then
    srun bash -c "echo 'imagenet not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -P -p 16 /fs/vulcan-datasets/imagenet /scratch0/mgwillia/"
fi

INDICES=(0 1 2 3 4 5 6)
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
CHECKPOINTS_DIRS=("chkpts_knn_btwins" "chkpts_knn_dc" "chkpts_knn_moco" "chkpts_knn_sclr" "chkpts_knn_simsiam" "chkpts_knn_sup" "chkpts_knn_swav")
STATE_DICT_KEY_NAMES=("classy_state_dict" "network" "network" "classy_state_dict" "network" "network" "classy_state_dict")

for i in ${INDICES[@]}; do
    srun bash -c "hostname; python ./tools/run_distributed_engines.py \
                    config=benchmark/nearest_neighbor/eval_resnet_8gpu_in1k_kNN.yaml \
                    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/vulcanscratch/mgwillia/unsupervised-classification/backbones/${BACKBONES[$i]}.torch \
                    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=${STATE_DICT_KEY_NAMES[$i]} \
                    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.NUM_NODES=1 \
                    config.DATA.TRAIN.DATA_PATHS=['/scratch0/mgwillia/imagenet/train'] \
                    config.DATA.TEST.DATA_PATHS=['/scratch0/mgwillia/imagenet/val'] \
                    config.CHECKPOINT.DIR=/vulcanscratch/mgwillia/vissl/${CHECKPOINTS_DIRS[$i]} \
                    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64"
done
