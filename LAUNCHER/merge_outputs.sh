#! /usr/bin/env bash
cd ../output
files=$(ls -l | grep 000 | grep -oE "[0-9]+_[0-9]+\.mp4" | sort -g)

rm inputs.txt

for x in $files ; do
 echo "file '$x'" >> inputs.txt ;
done;

ffmpeg -f concat -i inputs.txt -c copy merged.mp4
