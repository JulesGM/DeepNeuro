# !/usr/bin/env bash
CODE='ipython ./main.py 2>&1 | grep -v "naming conventions" | grep -v "raw_data = Raw(file_path)$"'
echo $CODE
eval "$CODE"
