#!/bin/bash
#SBATCH --job-name=vicuna7b-test
#SBATCH --account=dinov99
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --mail-user=naihe@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=vicuna-%j.out
#SBATCH --error=vicuna-%j.err

module purge
module load python/3.11.5
source ~/venv_vicuna/bin/activate

# Cache location (safe default: your home; change later to Turbo if needed)
export HF_HOME=/home/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$TRANSFORMERS_CACHE"

python ~/vicuna_test/test_vicuna.py
