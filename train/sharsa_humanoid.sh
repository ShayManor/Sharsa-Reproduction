#!/bin/bash -l
#SBATCH --job-name=sharsa_p45_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/sharsa_p45_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/sharsa_p45_1b_%j.err

set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_DIR=/scratch/gautschi/manors/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p /scratch/gautschi/manors/logs

cd "$HR_DIR"

python main.py \
  --run_group=paper_sharsa_humanoid \
  --seed=0 \
  --env_name=humanoidmaze-giant-navigate-oraclerep-v0 \
  --dataset_dir="$BASE/humanoidmaze-giant-navigate-1b-v0" \
  --dataset_replace_interval=4000 \
  --agent=agents/sharsa.py \
  --agent.q_agg=mean \
  --agent.subgoal_steps=50 \
  --agent.actor_p_trajgoal=1.0 \
  --agent.actor_p_randomgoal=0.0 \
  --agent.actor_geom_sample=False
