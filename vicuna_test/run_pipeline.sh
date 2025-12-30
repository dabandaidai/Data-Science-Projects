#!/bin/bash
#SBATCH --job-name=vicuna-pipeline
#SBATCH --account=dinov0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --output=pipe-%j.out
#SBATCH --error=pipe-%j.err

module purge
module load python/3.11.5
source ~/venv_vicuna/bin/activate

# SAFE cache location
export HF_HOME=/scratch/dinov_root/dinov0/$USER/hf_cache
mkdir -p "$HF_HOME"

cd ~/vicuna_test
python pipeline.py \
  --input_csv data/input.csv \
  --output_csv out/column_min_max_summary.csv