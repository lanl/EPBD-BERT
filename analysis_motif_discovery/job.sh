#!/usr/bin/sh

#SBATCH --job-name=Com_Overlapp
#SBATCH --output=/usr/projects/pyDNA_EPBD/tf_dna_binding/analysis_motif_finding/slurm_logs/%j.out
#SBATCH --error=/usr/projects/pyDNA_EPBD/tf_dna_binding/analysis_motif_finding/slurm_logs/%j.err
##SBATCH --mail-user=<akabir@lanl.gov>  # <akabir4@gmu.edu>
##SBATCH --mail-type=BEGIN,END,FAIL

## gpu
##SBATCH --partition=gpu 
##SBATCH --account=y23_unsupgan_g 
##SBATCH --mem=16G

# cpu
#SBATCH --partition=standard
#SBATCH --account=t23_dna-epbd 
#SBATCH --mem=32G
#SBATCH --time=16:00:00 ##HH:MM:SS

## conda activate /usr/projects/pyDNA_EPBD/tf_dna_binding/.venvs/python311_conda_3
export PYTHONPATH=$PYTHONPATH:$(pwd)
python analysis_motif_finding/find_motifs.py