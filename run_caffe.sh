#!/bin/bash
sbatch -p gpu --gres=gpu:1 --time=07-00 --exclude=node019 --mem=5000 --out=/om/user/hyo/caffe/output/$1.out --error=/om/user/hyo/caffe/output/$1.out $1
