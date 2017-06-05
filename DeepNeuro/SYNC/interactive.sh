#! /usr/bin/env bash
qsub -I -l walltime=10:00:00 -l nodes=1:gpus=1:ppn=1 -A kif-392-aa -q gpu_1
