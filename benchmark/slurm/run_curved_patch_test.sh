#!/bin/bash
#SBATCH --job-name=tm_curved_p
#SBATCH --output=curved_patch_test.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --partition=short

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Date: $(date)"

julia --project="${PROJECT_DIR}" -e '
    import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()
'
echo ">>> Precompile done: $(date)"

julia -t ${SLURM_CPUS_PER_TASK:-16} \
    --project="${PROJECT_DIR}" \
    "${SCRIPT_DIR}/../run_curved_patch_test.jl"

echo ">>> Finished: $(date)"
