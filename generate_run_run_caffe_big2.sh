#!/bin/bash

EXT=$1 

cat << EOF
#!/bin/bash
sbatch -p gpu --gres=gpu:1 --time=07-00 --exclude=node019 --mem=10000 --out=/om/user/hyo/caffe/output/$EXT.out --error=/om/user/hyo/caffe/output/$EXT.out run_caffe/run_caffe_reg_${EXT}.sh
EOF
