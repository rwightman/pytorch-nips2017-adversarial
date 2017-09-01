#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

python python/run_cw_inspired.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --ensemble AdvInceptionResnetV2 Inceptionv3 Resnet34 \
  --ensemble_weights 1.0 1.0 0.885 \
  --checkpoint_paths adv_inception_resnet_v2.pth inception_v3_google-1a9a5a14.pth resnet34-333f7ec4.pth\
  --batch_size 16 \
  --n_iter 80 \
  --lr=0.08
