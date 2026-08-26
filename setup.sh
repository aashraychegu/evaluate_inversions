slurm_jobid_prpt () { echo ""; }
export -f slurm_jobid_prpt
source "/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env/bin/activate"
module restore compute_stress
alias queue="squeue -u aashrayc --format='%.18i %.9P %.75j %.8u %.10T %.10M %.10l %.6D %R'"
