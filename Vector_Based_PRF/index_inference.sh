#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=repbert-index
#SBATCH -n 1
#SBATCH --mem-per-cpu=60G
#SBATCH -o print_log.txt
#SBATCH -e error_log.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate vprf
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2

srun python3 repbert_index_embedding_generator.py --model_path ./RepBERT_Model --embedding_output ./data/trec_deep_2019/index --per_gpu_batch_size 128
