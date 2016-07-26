source "/home/julesgm/COCO/FAKE_SCRATCH/myenv/bin/activate"

SCRIPT=/home/julesgm/COCO/PSD_perf_exploration.py
echo "${ARGS?$@}"
python "$SCRIPT" ${ARGS?$@}

wait
