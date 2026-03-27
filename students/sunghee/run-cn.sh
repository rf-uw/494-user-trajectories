#!/bin/bash
# """
# ./run-cn.sh
# """

# Exit on error
set -e


if [ ! -d "communitynotes_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv communitynotes_env
else
    echo "Virtual environment already exists."
fi

# 2. Activate environment
echo "Activating environment..."
source communitynotes_env/bin/activate

# 3. Install requirements
echo "Installing requirements..."
pip install -r communitynotes/requirements.txt

echo "Environment setup complete."

# 1. Preprocess data
python main.py 2026-01-31
