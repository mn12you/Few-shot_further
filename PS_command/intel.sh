#!/bin/bash
#SBATCH -A ACD112176      # Account name/project number
#SBATCH -J Corr_index generate    # Job name
#SBATCH -p ct560             # Partition name
#SBATCH -c 10              # Number of cores per MPI task
#SBATCH -n 48               # Number of MPI tasks (i.e. processes)
#SBATCH -o TWCC_log/%j.out           # Path to the standard output file
#SBATCH -e TWCC_log/%j.err           # Path to the standard error ouput file
#SBATCH --mail-user=s1212mn@gmail.com   # email
#SBATCH --mail-type=BEGIN,END    



module purge
ml miniconda3
conda activate ECG_SHAP_39 #進入 conda 環境
python -u "./data_mod/Corr_US.py"
