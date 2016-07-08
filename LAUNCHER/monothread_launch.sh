#@IgnoreInspection BashAddShebang
#PBS -A kif-392-aa
#PBS -l walltime=10:00:00
#PBS -l nodes=1:gpus=1
source /home/julesgm/COCO/LAUNCHER/launcher_prep.sh 1>/dev/null
## We start the job

echo -e " -- Starting --------------------.\n"
"$PYTHON" /home/julesgm/COCO/main.py && echo ">>> SUCCESS"
echo -e " -- Done ------------------------.\n"
wait
