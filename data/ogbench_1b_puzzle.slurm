#!/bin/bash -l
#SBATCH --job-name=ogbench_1b_cube
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/ogbench_1b_cube_%j.out

set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate
VENV_PY=/scratch/gautschi/manors/venvs/ogbench-py311/bin/python

export BASE=/scratch/gautschi/manors/ogbench
export REPO_DIR="$BASE/ogbench"
export EXPERT_DIR="$REPO_DIR/experts"
export PYTHONPATH="$REPO_DIR/impls:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p /scratch/gautschi/manors/logs
mkdir -p "$BASE/cube-octuple-play-1b-v0"
mkdir -p "$BASE/puzzle-4x5-play-1b-v0"
mkdir -p "$BASE/puzzle-4x6-play-1b-v0"
mkdir -p "$BASE/humanoidmaze-giant-navigate-1b-v0"

cd "$REPO_DIR"

if [[ ! -d "$EXPERT_DIR/humanoid" ]]; then
  echo "Missing $EXPERT_DIR/humanoid"
  exit 1
fi

trap 'echo "Stopping job, killing workers"; jobs -pr | xargs -r kill; wait; exit 1' INT TERM

MANIP_WORKERS=32
HUMANOID_WORKERS=16

run_manip_dataset() {
  local env_name="$1"
  local out_dir="$2"
  local prefix="$3"
  local max_seed="$4"
  local max_steps="$5"
  local workers="$6"

  local running=0

  echo "Starting $env_name -> $out_dir with $workers workers"

  for i in $(seq -w 0 "$max_seed"); do
    local out="$out_dir/$prefix-$i.npz"

    if [[ -s "$out" ]]; then
      echo "Skipping existing $out"
      continue
    fi

    "$VENV_PY" data_gen_scripts/generate_manipspace.py \
      --seed="$i" \
      --env_name="$env_name" \
      --save_path="$out" \
      --num_episodes=1000 \
      --max_episode_steps="$max_steps" \
      --dataset_type=play \
      > /scratch/gautschi/manors/logs/${prefix}_${i}.out 2>&1 &

    ((running+=1))

    if (( running >= workers )); then
      wait -n
      ((running-=1))
    fi
  done

  while (( running > 0 )); do
    wait -n
    ((running-=1))
  done

  echo "Finished $env_name"
}

run_humanoid_dataset() {
  local env_name="$1"
  local out_dir="$2"
  local prefix="$3"
  local max_seed="$4"
  local max_steps="$5"
  local workers="$6"

  local running=0

  echo "Starting $env_name -> $out_dir with $workers workers"

  for i in $(seq -w 0 "$max_seed"); do
    local out="$out_dir/$prefix-$i.npz"

    if [[ -s "$out" ]]; then
      echo "Skipping existing $out"
      continue
    fi

    "$VENV_PY" data_gen_scripts/generate_locomaze.py \
      --seed="$i" \
      --env_name="$env_name" \
      --save_path="$out" \
      --num_episodes=1000 \
      --max_episode_steps="$max_steps" \
      --dataset_type=navigate \
      --restore_path="$EXPERT_DIR/humanoid" \
      --restore_epoch=40000000 \
      > /scratch/gautschi/manors/logs/${prefix}_${i}.out 2>&1 &

    ((running+=1))

    if (( running >= workers )); then
      wait -n
      ((running-=1))
    fi
  done

  while (( running > 0 )); do
    wait -n
    ((running-=1))
  done

  echo "Finished $env_name"
}

#run_manip_dataset \
#  cube-octuple-v0 \
#  "$BASE/cube-octuple-play-1b-v0" \
#  cube-octuple-play-v0 \
#  999 \
#  1001 \
#  "$MANIP_WORKERS"

# run_manip_dataset \
#   puzzle-4x5-v0 \
#   "$BASE/puzzle-4x5-play-1b-v0" \
#   puzzle-4x5-play-v0 \
#   999 \
#   1001 \
#   "$MANIP_WORKERS"

 run_manip_dataset \
   puzzle-4x6-v0 \
   "$BASE/puzzle-4x6-play-1b-v0" \
   puzzle-4x6-play-v0 \
   999 \
   1001 \
   "$MANIP_WORKERS"

# run_humanoid_dataset \
#   humanoidmaze-giant-v0 \
#   "$BASE/humanoidmaze-giant-navigate-1b-v0" \
#   humanoidmaze-giant-navigate-v0 \
#   249 \
#   4001 \
#   "$HUMANOID_WORKERS"

echo "Selected 1B dataset generation complete."
