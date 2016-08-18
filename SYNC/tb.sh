#!/usr/bin/env bash

LOGDIR=$1
if [[ -z $1 ]] ; then
    echo -e "Haven't received a logdir, the first argument. We need a logdir:\n[ls /home/julesgm/COCO/saves/tf_summaries]"
    ssh helios ls -d -1 '~/COCO/saves/tf_summaries/*'
    ssh helios ls -d -1 '~/COCO/saves/*'
    exit
fi

d_REMOTE_PORT=9292
if [[ -n $2 ]] ; then
    REMOTE_PORT="$2";
    echo -e '$TARGET'" = $2\n"
else
    echo -e 'Defaulting to $REMOTE_PORT='"$d_REMOTE_PORT"
    REMOTE_PORT=$d_REMOTE_PORT
fi


d_TARGET=helios
if [[ -n $3 ]] ; then
    TARGET="$3";
    echo -e '$TARGET'" = $3\n"
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
#ssh helios -R $REMOTE_PORT:127.0.0.1:6006 "$CMD" -o TCPKeepAlive=yes
#ssh helios -R 8888:127.0.0.1:6006 -o TCPKeepAlive=yes
