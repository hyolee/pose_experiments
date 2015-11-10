#!/bin/bash

model=$1
nClass=$2
EXT="${model}_class${nClass}"
lr=$3

sh generate_train_val_${model}.sh $nClass > train_val/train_val_${EXT}.prototxt
sh generate_solver.sh $EXT $lr > solver/solver_${EXT}_lr${lr}.prototxt
sh generate_run_caffe.sh $EXT $lr > run_caffe/run_caffe_${EXT}_lr${lr}.sh
sh generate_run_run_caffe.sh ${EXT}_lr${lr} > run_caffe/run_run_caffe_${EXT}_lr${lr}.sh
