#!/usr/bin/sh

#SBATCH --job-name=Com_Overlapp
#SBATCH --output=/usr/projects/pyDNA_EPBD/tf_dna_binding/slurm_logs/out-%N-%j.output
#SBATCH --error=/usr/projects/pyDNA_EPBD/tf_dna_binding/slurm_logs/err-%N-%j.error
##SBATCH --mail-user=<akabir@lanl.gov>  # <akabir4@gmu.edu>
##SBATCH --mail-type=BEGIN,END,FAIL

## gpu
##SBATCH --partition=gpu 
##SBATCH --account=y23_unsupgan_g 
##SBATCH --mem=16G

# cpu
#SBATCH --partition=standard
#SBATCH --account=t23_dna-epbd 
#SBATCH --mem=8G
#SBATCH --time=12:00:00 ##HH:MM:SS

#SBATCH --array=0-114

bash data_preprocessing/2.2_compute_overlappings.sh