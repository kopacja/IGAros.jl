#!/bin/bash
#
# SLURM single-node thread-scaling sweep for IGAros stiffness assembly.
# Runs all thread counts sequentially on ONE node, so timing differences
# reflect only threading, not hardware variation.
#
# Usage:  cd IGAros/benchmark && sbatch bench_assembly_sweep.sh
#
# Results land in bench_assembly_sweep.out

#SBATCH --job-name=iga_sweep
#SBATCH --output=bench_assembly_sweep.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --partition=express
#SBATCH --exclusive

BENCH_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PROJECT_DIR="$(cd "${BENCH_DIR}/.." && pwd)"
JULIA_BIN="julia"

echo "Host: $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Benchmark dir: ${BENCH_DIR}"
echo "Date: $(date)"
echo ""

# Instantiate once before the sweep
${JULIA_BIN} --project="${PROJECT_DIR}" -e '
    import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()
'

for NT in 1 2 4 8 16 32 48 96; do
    echo ""
    echo "================================================================"
    echo "  Thread count: ${NT}"
    echo "================================================================"
    ${JULIA_BIN} -t ${NT} --project="${PROJECT_DIR}" "${BENCH_DIR}/bench_assembly.jl"
done
