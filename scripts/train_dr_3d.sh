#!/bin/bash
# Baseline 3D ResNet experiment on the DR task with OCT B-scans.
# Usage:  ./scripts/train_dr_3d.sh
# Override the dataset location if needed:  DATASET_DIR=/path/to/FairVision ./scripts/train_dr_3d.sh
DATASET_DIR=${DATASET_DIR:-/home/jupyter-vshein/data/harvard/FairVision/}
RESULT_DIR=${RESULT_DIR:-.}

LR=5e-5
NUM_EPOCH=50
BATCH_SIZE=2
MODALITY_TYPE='oct_bscans_3d'
ATTRIBUTE_TYPE=( race gender hispanic ) # race|gender|hispanic
TASK=cls
LOSS_TYPE=bce

MODEL_TYPE=resnet18
CONV_TYPE=Conv3d

PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_3D_baseline.csv
python ./scripts/train_dr_fair_3d.py \
		--data_dir ${DATASET_DIR}/DR/ \
		--result_dir ${RESULT_DIR}/results_3D_baseline_1/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${MODEL_TYPE}_${CONV_TYPE}_3D_baseline_lr${LR} \
		--model_type ${MODEL_TYPE} \
		--image_size 200 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch_size ${BATCH_SIZE} \
		--task ${TASK} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE} \
		--conv_type ${CONV_TYPE}
