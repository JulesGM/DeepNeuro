#@IgnoreInspection BashAddShebang
#PBS -A kif-392-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=1

source /home/julesgm/COCO/LAUNCHER/launcher_prep.sh 1>/dev/null

echo ">>>>>>>>>> \$1 is '$julesID'"
"$PYTHON" /home/julesgm/COCO/main.py "$julesID" 800
wait
