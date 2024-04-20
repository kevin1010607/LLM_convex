#!/bin/bash
#SBATCH --job-name=llm_test      ## job name
#SBATCH --nodes=1                ## request 1 nodes
#SBATCH --ntasks-per-node=1      ## run 1 srun task per node
#SBATCH --cpus-per-task=32       ## allocate 32 CPUs per srun task
#SBATCH --gres=gpu:8             ## request 8 GPUs per node
#SBATCH --time=20:00:00          ## run for a maximum of 20 hours
#SBATCH --account=ACD110018      ## PROJECT_ID, please fill in your project ID (e.g., XXX)
#SBATCH --partition=gp1d         ## gtest is for testing; you can change it to gp1d (1-day run), gp2d (2-day run), gp4d (4-day run), etc.
#SBATCH -o %j.out                ## Path to the standard output file
#SBATCH -e %j.err                ## Path to the standard error output file
#SBATCH --mail-user={your email} 
#SBATCH --mail-type=BEGIN,END

# Load module
module purge
module load pkg/Anaconda3 cuda/11.7 compiler/gcc/11.2.0
module list

# Activate conda
which conda
conda activate alpaca

# Run inference
python simple_generate.py \
    --base_model meta-llama/Llama-2-7b-hf
