#!/bin/bash -l
#SBATCH --job-name=regresson_pearson

 
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-120:00:00
#SBATCH --partition compute
#SBATCH --output=/home/sheczko/logs/%j.out

 
module add singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"

singularity exec /ptmp/containers/datascience-python_latest.sif  python /home/sheczko/code1/MastersThesis/regresson_launcher_pearson.py


exit 0
