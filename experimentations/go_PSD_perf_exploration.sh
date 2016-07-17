#!/usr/bin/env bash
d_TARGET=helios


if [[ -n $1 ]] ; then
    TARGET=$1;
    echo '$TARGET'" = $1"
else
    echo 'Defaulting to $TARGET = '"$d_TARGET"
    TARGET=$d_TARGET
fi


#################################
# Setup of target specific code
#################################

if [[ "$TARGET" == "guillimin" ]] ; then
    LAUNCHER=qsub ;
    PY_MODULE="python/2.7.9"
    DATA_PATH='$SCRATCH/aut_gamma/'
elif [[ "$TARGET" == "helios" ]] ; then
    LAUNCHER=msub ;
    PY_MODULE="apps/python/2.7.10"
    DATA_PATH='/home/julesgm/aut_gamma'
else
    echo "Please specify a valid TARGET. Got \"$TARGET\"."
    exit
fi


#################################
#
#################################

echo "<rsync>"
/home/jules/Documents/COCO_NEURO/SYNC/to "$TARGET" >/dev/null
echo "</rsync>"


VENV=/home/julesgm/COCO/FAKE_SCRATCH/myenv/bin/activate
CMD="
cd /home/julesgm/COCO/experimentations;
echo '<module load python>'
module load ${PY_MODULE}
echo '</module load python>'
echo '<source VENV>'
source $VENV
echo '</source VENV>'
echo '<python exec>'
python gw_PSD_perf_exploration.py --launcher $LAUNCHER --host $TARGET --data_path $DATA_PATH
echo '</python exec>'
"

echo "<ssh>"
#cat "$CMD"
ssh "$TARGET" "$CMD"
echo "</ssh>"
