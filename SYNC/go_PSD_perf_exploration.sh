#!/usr/bin/env bash
# the snippet
#      | sed 's/^/\t/'
# is to indent the output of commands

d_TARGET=helios
d_JOBTYPE="linear"

C_reset='\033[00m'
C_white='\033[01;37m'
C_light_blue='\033[01;34m'
C_light_cyan='\033[01;36m'
C_dark_blue='\033[02;34m'

LB=${C_light_cyan}
DB=${C_light_blue}


if [[ -n $1 ]] ; then
    TARGET="$1";
    echo -e '$TARGET'" = $1\n"
else
    echo -e 'Defaulting to $TARGET='"$d_TARGET"
    TARGET=$d_TARGET
fi

if [[ -n $2 ]] ; then
    JOBTYPE="$2";
    echo -e '$JOBTYPE'" = $2\n"
else
    echo -e 'Defaulting to $JOBTYPE='"$d_JOBTYPE"
    TARGET=$d_JOBTYPE
fi





# echo  -e "${LB}<pylint -E ./*.py>${C_reset}\n"
# pylint -E ~/COCO/*.py  2>&1 | sed 's/^/\t/'
# echo  -e "\n${LB}</pylint -E ./*.py>${C_reset}"


echo  -e "${LB}<generating requirements.txt with sfood>${C_reset}\n"
sfood ~/COCO/ -e 2>/dev/null | requirements_generator 2>/dev/null | tee requirements.txt | sed 's/^/\t/'
echo -e "\n$LB</generating requirements.txt with sfood>$C_reset"
echo -e "$LB<to::rsync>$C_reset\n"
to $TARGET | sed 's/^/\t/'
echo -e "\n$LB</to::rsync>$C_reset"


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


REM="${DB}remote:${C_reset} "

VENV=/home/julesgm/COCO/FAKE_SCRATCH/myenv/bin/activate
CMD="
cd /home/julesgm/COCO/
echo -e '\n'
echo -e \"${REM}${LB}<module load python>$C_reset\"
module load ${PY_MODULE} | sed 's/^/\t/'
echo -e \"${REM}${LB}</module load python>${C_reset}\"
echo -e \"${REM}${LB}<source VENV>$C_reset\"
source $VENV | sed 's/^/\t/'
echo -e \"${REM}${LB}</source VENV>$C_reset\"
echo -e \"${REM}${LB}<pip install -r requirements.txt -U>$C_reset\"
pip install -r requirements.txt -U 1>/dev/null
echo -e \"${REM}${LB}</pip install -r requirements.txt -U>$C_reset\"
echo -e \"${REM}${LB}<python exec>$C_reset\n\"
python gw_PSD_perf_exploration.py --host $TARGET --data_path $DATA_PATH --job_type $JOBTYPE 2>&1 | sed 's/^/\t/'
echo -e \"${REM}${LB}</python exec>$C_reset\"
echo -e \"\n\"
"

echo -e "$LB<ssh>$C_reset"
ssh "$TARGET" "$CMD" | sed 's/^/\t/'
echo -e "$LB</ssh>$C_reset"
