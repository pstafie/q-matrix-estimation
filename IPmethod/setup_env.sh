#!/bin/bash
# =============================================================================
# setup_env.sh
# Run this ONCE on Great Lakes before submitting your job.
# It creates a virtual environment and installs all dependencies.
#
# Usage:
#   bash setup_env.sh
# =============================================================================

set -e  # exit on any error

echo "=== Loading modules ==="
module load python/3.11.5
module load gurobi/11.0.3

echo "=== Creating virtual environment ==="
python3 -m venv ~/dina_mip_env

echo "=== Activating virtual environment ==="
source ~/dina_mip_env/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing dependencies ==="
pip install numpy scipy python-sat joblib

echo "=== Installing dina-qip package ==="
# Assumes you uploaded dina-qip to your home directory at ~/dina-qip/dina-qip
pip install -e ~/dina-qip/dina-qip

echo ""
echo "=== Setup complete! ==="
echo "Virtual environment is at: ~/dina_mip_env"
echo "You can now submit your job with: sbatch run_mip_job.sh"
