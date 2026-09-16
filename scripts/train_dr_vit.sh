#!/bin/bash
# Baseline (no fair identity scaling) ViT-B experiment on the DR task with SLO fundus images.
# Usage:  ./scripts/train_dr_vit.sh
# Override the dataset location if needed:  DATASET_DIR=/path/to/FairVision ./scripts/train_dr_vit.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
DATASET_DIR=${DATASET_DIR:-${REPO_ROOT}/data}
RESULT_DIR=${RESULT_DIR:-.}
MODEL_TYPE=( ViT-B ) # Options: efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=( race gender hispanic ) # Options: race | gender | hispanic

# Args (slo_fundus)
VIT_WEIGHTS=imagenet
BATCH_SIZE=64
BLR=5e-4
WD=0.01
LD=0.55
DP=0.1
EXP_NAME=${VIT_WEIGHTS}_slo_fundus

PERF_FILE=DR_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python scripts/train_dr_fair.py \
        --epochs 50 \
        --batch_size ${BATCH_SIZE} \
        --blr ${BLR} \
        --min_lr 1e-6 \
        --warmup_epochs 5 \
        --weight_decay ${WD} \
        --layer_decay ${LD} \
        --drop_path ${DP} \
        --data_dir ${DATASET_DIR}/DR/ \
        --result_dir ${RESULT_DIR}/results/DR_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE} \
        --model_type ${MODEL_TYPE} \
        --modality_types ${MODALITY_TYPE} \
        --perf_file ${PERF_FILE} \
        --vit_weights ${VIT_WEIGHTS} \
        --attribute_type ${ATTRIBUTE_TYPE}
