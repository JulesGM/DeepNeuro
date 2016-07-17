source /home/julesgm/.bashrc
export TERM="xterm-256color"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre2/software-gpu/compilers/pgi/15.10/linux86-64/2015/cuda/7.5/lib64/"

export ENV="/home/julesgm/COCO/FAKE_SCRATCH/"

source "$ENV/myenv/bin/activate"
if [[ $? -ne 0 ]] ; then
	source "/home/julesgm/COCO/LAUNCHER/env_prep.sh"
fi ;

$PYTHON -c "import tensorflow"
if [[ $? -ne 0 ]] ; then
	source "/home/julesgm/COCO/LAUNCHER/env_prep.sh"
fi ;

# This is just so we have a scratch when we debug on the gateway node
if [[ -z "$LSCRATCH" ]] ; then
	export LSCRATCH="/home/julesgm/COCO/FAKE_SCRATCH/"
fi;
