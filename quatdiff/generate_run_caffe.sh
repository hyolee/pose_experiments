#!/bin/bash

EXT=$1
lr=$2

cat << EOF
#!/bin/bash
/om/user/hyo/src/caffe/build/tools/caffe train -solver /om/user/hyo/caffe/quatdiff/solver/solver_${EXT}_lr${lr}.prototxt
EOF
#sbatch -p gpu --gres=gpu:1 --time=07-00 --exclude=node019 --mem=5000 --job-name=${EXT}_lr${lr} --out=/om/user/hyo/caffe/output/${EXT}_lr${lr}.out --error=/om/user/hyo/caffe/output/${EXT}_lr${lr}.out /om/user/hyo/src/caffe/build/tools/caffe train -solver /om/user/hyo/caffe/solver/solver_reg_${EXT}_lr${lr}.prototxt
