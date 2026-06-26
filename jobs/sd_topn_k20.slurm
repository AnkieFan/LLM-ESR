#!/bin/bash
#SBATCH --job-name=sd_topn_k20
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --array=42-44
#SBATCH --output=%x_seed%a_%A.out
#SBATCH --error=%x_seed%a_%A.err

module load 2024 Anaconda3/2024.06-1
source /sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate llm
cd $HOME/LLM-ESR

SEED=$SLURM_ARRAY_TASK_ID
echo "[$(date)] sd_topn_k20 llmesr_sasrec seed=$SEED"
python main.py --dataset yelp --model_name llmesr_sasrec \
  --hidden_size 64 --train_batch_size 128 --max_len 200 --gpu_id 0 \
  --num_workers 8 --num_train_epochs 200 --seed $SEED --check_path sd_topn_k20_s$SEED \
  --patience 20 --ts_user 12 --ts_item 13 --freeze --log \
  --user_sim_func kd --alpha 0.1 --use_cross_att --sim_user_num 20
echo "[$(date)] sd_topn_k20 seed=$SEED done (exit $?)"
