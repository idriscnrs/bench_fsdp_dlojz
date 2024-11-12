#!/bin/bash
#SBATCH --job-name=bench_h100_cap
#SBATCH --output=slurm/log/%j.out 
#SBATCH --error=slurm/log/%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread 
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --account=sos@h100


## load module 
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0


## launch script on every task 
set -x
time srun python -u fsdp.py --batch-size 16 --num-workers 2 --seq-len 512 --test --compile
date
