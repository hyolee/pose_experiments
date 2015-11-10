#!/bin/bash

nClass=$1
dim=$2
EXT="big2_class${nClass}_${dim}d"
lr=$3

sh generate_train_val_big2_${dim}d.sh $nClass > train_val/train_val_reg_${EXT}.prototxt
sh generate_solver_short_${dim}d.sh $EXT $lr > solver/solver_reg_${EXT}_lr${lr}.prototxt
sh generate_run_caffe.sh $EXT $lr > run_caffe/run_caffe_reg_${EXT}_lr${lr}.sh
sh generate_run_run_caffe_big2.sh ${EXT}_lr${lr} > run_caffe/run_run_caffe_reg_${EXT}_lr${lr}.sh
