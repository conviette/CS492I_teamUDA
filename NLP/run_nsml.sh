#!/usr/bin/env bash

nsml run \
-d korquad-open-ldbd3 \
-g 2 -c 1 -e run_squad.py \
-a "--do_train \
 --do_eval \
 --data_dir train \
 --num_train_epochs 1 \
 --per_gpu_train_batch_size 24 \
 --per_gpu_eval_batch_size 24 \
 --output_dir output \
 --overwrite_output_dir \
 --version_2_with_negative"
