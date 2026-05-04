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

# Check for Hermes API key (needed for server-side session continuity)
if [ -z "$HERMES_API_KEY" ]; then
    echo "⚠️  HERMES_API_KEY not set — server-side session disabled."
    echo "   Set it: export HERMES_API_KEY=robot-bridge-key"
fi

# Start the server
echo "Starting Robot Bridge..."
python -m src.main
