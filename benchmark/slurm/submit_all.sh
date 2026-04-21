#!/bin/bash
# Submit all 6 benchmark jobs to the SLURM cluster.
# Usage:  cd ~/IGAros/benchmark/slurm && bash submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for ex in flat_patch_test plate_with_hole cylinders curved_patch_test bending_beam pressurized_sphere; do
    echo "Submitting: $ex"
    cd "${SCRIPT_DIR}" && sbatch "run_${ex}.sh"
done

echo ""
echo "All jobs submitted. Check with: squeue -u $(whoami)"
