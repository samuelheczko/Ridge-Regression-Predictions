#!/bin/bash -l
#SBATCH --job-name=feature importance
 
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-24:00:00
#SBATCH --partition compute
#SBATCH --output=/home/sheczko/logs/%j.out

 
module add singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"

singularity exec /ptmp/containers/datascience-python_latest.sif  python /home/sheczko/code1/MastersThesis/importance_finder_tangent.py


exit 0
