#!/bin/bash -l
#SBATCH --job-name=finegym_clips_extraction    # Job name for identification
#SBATCH --output=/proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/%j.out # Standard output and error log (%j is the job ID)
#SBATCH --error=/proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/%j.err  # Separate error log
#SBATCH --nodes=1                     # Request a single node
#SBATCH --ntasks-per-node=1           # Run a single task
#SBATCH --cpus-per-task=4             # Request 4 CPUs (moviepy can use them)
#SBATCH --mem=32G                     # Request 16GB of memory
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
nvidia-smi

source /home/x_ashsi/miniconda3/etc/profile.d/conda.sh
conda activate mlv1

# --- Job Setup ---
echo "Job started on ${hostname} at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"

export PROJECT_PATH=/home/x_ashsi/Research/Projects/VideoReprSSL
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH


# ------------------------------------------------------------
echo "Create directory for log"
CURRENTDATE=`date + "%Y-%m-%d"`
echo $CURRENTDATE
PATHLOG="proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/"
echo $PATHLOG
output_file="${PATHLOG}/${SLURM_JOB_ID}_logs.txt"

# ------------------------------------------------------------
cd $PROJECT_PATH/video_embeddings/finegym
which python
time python extract_video_clips.py config.ini >> $output_file

echo "Job finished at $(date)"
