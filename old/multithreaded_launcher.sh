#@IgnoreInspection BashAddShebang

#PBS -A kif-392-aa
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=4

source ./launcher_prep.sh

## We start the job
echo -e " -- Starting --------------------.\n"

for i in $(seq 1 20) ; do
    $PYTHON /home/julesgm/COCO/main.py $i &
done ;
echo -e " -- Done ------------------------.\n"
wait
