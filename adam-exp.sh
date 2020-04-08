#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=gpu_devices
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=34:15:00
#SBATCH --output=log_exp.out
#SBATCH --mail-user=graduationprojectplease@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10

# list-gpu-devices/list.sh (Slurm submission script)

python -u nutshell/occ_sep_adam_exp.py
