
#!/bin/bash -l
#SBATCH --job-name=test_sbatch_script
 
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-24:00:00
#SBATCH --partition compute
 
module add singularity
export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
 
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}
 
cp ${SLURM_SUBMIT_DIR}/.py ${SCRATCH_DIRECTORY}


singularity exec /ptmp/containers/fmriprep_latest.sif  python /home/sheczko/code1/MastersThesis/FS_MS.py
cp -r ${SCRATCH_DIRECTORY}/output ${SLURM_SUBMIT_DIR}/${SLURM_JOBID}
 
exit 0