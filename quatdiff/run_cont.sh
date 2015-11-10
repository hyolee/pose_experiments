#!/bin/bash

ext="1_class"$1
jobID=$2

cp output/${ext}.out output/${ext}_0.out
sbatch --dependency=afterany:${jobID} run_caffe/run_run_caffe_${ext}.sh
