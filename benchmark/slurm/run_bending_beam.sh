#!/bin/bash
#SBATCH --job-name=tm_bending_
#SBATCH --output=bending_beam.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --partition=express

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Date: $(date)"

julia --project="${PROJECT_DIR}" -e '
    import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()
'
echo ">>> Precompile done: $(date)"

julia -t ${SLURM_CPUS_PER_TASK:-32} \
    --project="${PROJECT_DIR}" \
    "${SCRIPT_DIR}/../run_bending_beam.jl"

echo ">>> Finished: $(date)"
