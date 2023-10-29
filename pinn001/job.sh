#!/bin/bash
#SBATCH -J pinn_001
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --nodelist=g2
#SBATCH --mail-user=lucas97pancotto@gmail.com
#SBATCH --mail-type=end

ml cuda/11.1
ml python/3.7.9
export CUDA_VISIBLE_DEVICES="1"
python3 -u run_pinn.py
