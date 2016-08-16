#!/usr/bin/env bash

d_TARGET=helios
if [[ -n $1 ]] ; then
    TARGET="$1";
    echo -e '$TARGET'" = $1\n"
else
    echo -e 'Defaulting to $TARGET='"$d_TARGET"
    TARGET=$d_TARGET
fi

#################################
# Setup of target specific code
#################################
if [[ "$TARGET" == "guillimin" ]] ; then
    PY_MODULE="python/2.7.9"
    DATA_PATH='$SCRATCH/aut_gamma/'
elif [[ "$TARGET" == "helios" ]] ; then
    PY_MODULE="apps/python/2.7.10"
    DATA_PATH='/home/julesgm/aut_gamma'
else
    echo "Please specify a valid TARGET. Got \"$TARGET\"."
    exit
fi


REMOTE_PORT=9999

VENV=/home/julesgm/COCO/FAKE_SCRATCH/myenv/bin/activate
CMD="
source /home/julesgm/.bashrc
source /home/julesgm/.bash_profile
killall -9 tensorboard
cd /home/julesgm/COCO/
module load ${PY_MODULE}
source ${VENV}
tensorboard --logdir='/home/julesgm/COCO/saves/tf_summaries/' --port $REMOTE_PORT
"

# generate a random name (poorly, lol.)
pipe=$(mktemp)
# remove the file created at that dir
rm $pipe
mkfifo $pipe

if [[ "$?" -eq 1 ]] ; then
  echo "mkfifo failed. exiting"
  exit 1
fi

cleanup () {
  ssh helios killall tensorboard
}
trap cleanup EXIT

( sleep 20 ; google-chrome 127.0.0.1:6006 ) &
ssh helios -L 6006:127.0.0.1:$REMOTE_PORT "$CMD"
