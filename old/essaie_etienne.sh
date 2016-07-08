#@IgnoreInspection BashAddShebang
#PBS -A kif-392-aa
#PBS -l walltime=10:00:00
#PBS -l nodes=1:gpus=1

source ./launcher_prep.sh

## We start the job
echo -e " -- Starting --------------------.\n"
$LSCRATCH/myenv/bin/python /home/julesgm/COCO/main.py
echo -e " -- Done ------------------------.\n"
wait
