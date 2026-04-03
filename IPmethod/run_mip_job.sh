#!/bin/bash
# =============================================================================
# run_mip_job.sh — SLURM job script for Great Lakes (U of M)
#
# Submit with:
#   sbatch run_mip_job.sh
#
# Monitor with:
#   squeue -u stafie
#
# Check output live:
#   tail -f ~/dina_mip_study/logs/mip_sim_JOBID.out
# =============================================================================

#SBATCH --job-name=dina_mip_sim
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --time=12:00:00
#SBATCH --output=/home/stafie/dina_mip_study/logs/mip_sim_%j.out
#SBATCH --error=/home/stafie/dina_mip_study/logs/mip_sim_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stafie@umich.edu

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p /home/stafie/dina_mip_study/logs
cd /home/stafie/dina_mip_study

module load python3.11-anaconda/2024.02
module load gurobi/11.0.3

source ~/dina_mip_env/bin/activate

echo "Starting simulation..."
python run_mip_simulation.py

echo "Job finished: $(date)"
