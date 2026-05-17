#!/bin/bash

set -e

echo "Starting startup sequence..."

python API/scripts/download_database.py

python API/scripts/pull_model.py

echo "Launching API..."

uvicorn API.scripts.main:app --host 0.0.0.0 --port 8000
