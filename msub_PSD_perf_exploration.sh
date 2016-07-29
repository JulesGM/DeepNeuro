source "/home/julesgm/COCO/FAKE_SCRATCH/myenv/bin/activate"

SCRIPT=/home/julesgm/COCO/actual_work.py
echo "${ARGS?$@}"
python "$SCRIPT" ${ARGS?$@}

wait
