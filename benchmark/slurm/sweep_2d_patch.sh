#!/bin/bash
#SBATCH --job-name=sweep2d
#SBATCH --output=sweep_2d_patch.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --partition=express

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Host: $(hostname), Date: $(date)"

julia --project="${PROJECT_DIR}" -e '
    import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()
'

julia -t ${SLURM_CPUS_PER_TASK:-8} \
    --project="${PROJECT_DIR}" \
    "${SCRIPT_DIR}/../sweep_2d_patch.jl"

echo ">>> Finished: $(date)"
