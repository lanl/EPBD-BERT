#!/usr/bin/sh

#SBATCH --job-name=hum
#SBATCH --output=/usr/projects/pyDNA_EPBD/tf_dna_binding/slurm_logs/hum-%j.out
#SBATCH --error=/usr/projects/pyDNA_EPBD/tf_dna_binding/slurm_logs/hum-%j.err
#SBATCH --mail-user=<akabir@lanl.gov>
#SBATCH --mail-type=BEGIN,END,FAIL

## gpu
##SBATCH --partition=gpu 
##SBATCH --account=y23_unsupgan_g 
##SBATCH --mem=16G

# cpu
#SBATCH --partition=standard
#SBATCH --account=t23_dna-epbd 
#SBATCH --mem=8G
#SBATCH --time=16:00:00 ##HH:MM:SS


#SBATCH --array=0-299 # max 300(0-299) can be requested

python -m pydna_epbd.run --config_filepath data/pydnaepbd_things/configs.txt