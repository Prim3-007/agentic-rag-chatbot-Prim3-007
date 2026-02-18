#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Use .venv python if exists
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD scripts/sanity_check.py