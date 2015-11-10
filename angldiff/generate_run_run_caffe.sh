#!/bin/bash

EXT=$1 

cat << EOF
#!/bin/bash
sbatch --job-name=$EXT -p gpu --gres=gpu:1 --time=07-00 --exclude=node019,node027 --mem=5000 --out=/om/user/hyo/caffe/angldiff/output/$EXT.out --error=/om/user/hyo/caffe/angldiff/output/$EXT.out run_caffe/run_caffe_${EXT}.sh
EOF
