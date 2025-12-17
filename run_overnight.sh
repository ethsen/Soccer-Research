#!/usr/bin/env bash
set -u  # error on unset vars (but don't stop on command failures)

mkdir -p overnight_runs logs

run() {
  local name="$1"; shift
  echo "=============================="
  echo "[START] $name  $(date)"
  echo "CMD: $*"
  echo "=============================="

  # Run command, capture exit code, always continue
  "$@" >"logs/${name}.out" 2>"logs/${name}.err"
  local code=$?

  if [ $code -eq 0 ]; then
    echo "[OK]   $name  $(date)"
  else
    echo "[FAIL] $name (exit=$code)  $(date)"
  fi
  echo "$name,$code,$(date)" >> logs/summary.csv
  echo
}

# clear summary for this run
echo "job,exit_code,finished_at" > logs/summary.csv

run PassMap    python fashionnet-train.py --data_root segmented_experiments/torch_data_masked --out_dir fn_day_runs/ --epochs 40 --arch fn-small
run fn_med_masked     python fashionnet-train.py --data_root segmented_experiments/torch_data_masked --out_dir fn_day_runs/ --epochs 40 --arch fn-medium

echo "All jobs done: $(date)"
echo "See logs/summary.csv and logs/*.out|*.err"
