# !/usr/bin/env bash
CODE='time ipython ./main.py 2>&1 | grep -v "naming conventions" | grep -v "data = fid.read(int(end - start))$"'
echo $CODE
eval "$CODE"
