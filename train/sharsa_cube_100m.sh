#!/bin/bash -l
#SBATCH --job-name=sharsa_cube_100m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/sharsa_cube_100m_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/sharsa_cube_100m_%j.err

set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_DIR=/scratch/gautschi/manors/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs
cd "$HR_DIR"

python main.py \
  --run_group=paper_sharsa_cube_100m \
  --seed=0 \
  --env_name=cube-octuple-play-oraclerep-v0 \
  --dataset_dir="$BASE/cube-octuple-play-100m-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=50 \
  --agent.actor_p_trajgoal=1.0 \
  --agent.actor_p_randomgoal=0.0 \
  --agent.actor_geom_sample=True \
  --video_episodes=0
