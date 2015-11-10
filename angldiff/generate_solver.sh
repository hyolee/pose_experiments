#!/bin/sh

EXT=$1
lr=$2

cat << EOF
net: "/om/user/hyo/caffe/angldiff/train_val/train_val_${EXT}.prototxt"
test_iter: 1000
test_interval: 1000
base_lr: 1e-${lr}
lr_policy: "step"
gamma: 0.1
stepsize: 200000
display: 1
max_iter: 900000
momentum: 0.9
weight_decay: 0.0005
snapshot: 10000
snapshot_prefix: "/om/user/hyo/caffe/angldiff/snapshot/${EXT}_lr${lr}"
solver_mode: GPU
EOF
