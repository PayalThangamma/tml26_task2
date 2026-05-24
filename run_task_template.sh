#!/bin/bash
set -e

cd $HOME/tml26_task2

echo "Current directory:"
pwd

echo "Checking files:"
find target_model -name "*.safetensors" | wc -l
find suspect_models -name "*.safetensors" | wc -l
du -sh .

echo "Installing small missing packages:"
python -m pip install pandas safetensors requests

echo "Running task_template.py"
python task_template.py

echo "Checking outputs:"
ls -lh submission.csv debug_scores.csv
wc -l submission.csv
head submission.csv
