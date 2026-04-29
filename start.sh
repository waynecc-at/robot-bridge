#!/bin/bash
# Start Robot Bridge

cd "$(dirname "$0")"

# Check if Python venv exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv and install dependencies
source .venv/bin/activate
pip install -q -e .

# Start the server
echo "Starting Robot Bridge..."
python -m src.main
