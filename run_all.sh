#!/usr/bin/env bash
# Run every config and rebuild docs/index.html.
#   Daphnet:  data_dir in configs/daphnet.yaml (override with DAPHNET_DIR=/path/to/dataset_fog_release/dataset)
#   JIGSAWS:  a task whose data folder is missing is reported as "awaiting dataset access" and skipped.
# Saved Daphnet results that already come from the same config are NOT re-trained (--skip-if-reproduced);
# the script then only rebuilds their tables and figures.
set -euo pipefail
cd "$(dirname "$0")"

DAPHNET_DIR="${DAPHNET_DIR:-$(python -c "import yaml;print(yaml.safe_load(open('configs/daphnet.yaml'))['data_dir'])")}"

if [ -d "$DAPHNET_DIR" ] || [ -f results/daphnet/results.json ]; then
  python main.py --config configs/daphnet.yaml --data-dir "$DAPHNET_DIR" --skip-if-reproduced
  if [ -d results/daphnet/models ] || [ -f results/daphnet/robustness.json ]; then
    python main.py --config configs/daphnet.yaml --data-dir "$DAPHNET_DIR" --robustness --skip-if-reproduced
  fi
  if [ ! -f results/deployment.json ] && [ -d "$DAPHNET_DIR" ] && [ -d results/daphnet/models ]; then
    python export.py --data-dir "$DAPHNET_DIR"
  fi
else
  echo "[daphnet] $DAPHNET_DIR not found and no saved results: download Daphnet from the UCI repository first"
fi

for cfg in configs/jigsaws_*.yaml; do
  echo "== $cfg"
  python main.py --config "$cfg"
done

python make_report.py
