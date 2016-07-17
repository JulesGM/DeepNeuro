#!/usr/bin/env bash

if [[ -n $1 ]] ; then
    TARGET="$1" ;
else
    echo Defaulting to Helios
    TARGET=helios ;
fi

echo \$1: \""$1"\"
echo Target: \""$TARGET"\"

rsync --partial --progress -r "$TARGET":"~/COCO/*" /home/jules/Documents/COCO_NEURO/ --exclude FAKE_SCRATCH --exclude mne_inst --exclude output


