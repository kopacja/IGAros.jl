#!/bin/bash
# submit_flat_patch_3d.sh — Submit all 3D flat patch test benchmark jobs to SLURM
#
# Creates a shared results directory and submits three independent jobs:
#   1. factorial_moments  (30 min, 16 GB)
#   2. eps_sweep          (1 hour, 32 GB)
#   3. nquad_sweep        (1.5 hours, 16 GB)

set -euo pipefail

RESULTS_DIR="${HOME}/Projects/twin_mortar/results/flat_patch_test_3d/$(date +%Y-%m-%d)_benchmark"
mkdir -p "${RESULTS_DIR}"
export TM_RESULTS_DIR="${RESULTS_DIR}"

echo "Results directory: ${RESULTS_DIR}"
echo ""

# ── Job 1: Factorial + Moments ────────────────────────────────────────────────
jid1=$(sbatch --export=ALL <<'SLURM' | awk '{print $NF}'
#!/bin/bash
#SBATCH --job-name=tm3d_fact
#SBATCH --output=factorial_moments_3d.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --partition=express

PROJECT_DIR="${HOME}/Projects/twin_mortar/IGAros"

echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Date: $(date)"
echo "Results → ${TM_RESULTS_DIR}"

julia --project="${PROJECT_DIR}" -e 'import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()'
echo ">>> Precompile done: $(date)"

julia -t ${SLURM_CPUS_PER_TASK:-8} --project="${PROJECT_DIR}" "${PROJECT_DIR}/examples/run_factorial_moments_3d.jl"
echo ">>> Finished: $(date)"
SLURM
)
echo "Submitted factorial_moments:  job ${jid1}"

# ── Job 2: Epsilon sweep ──────────────────────────────────────────────────────
jid2=$(sbatch --export=ALL <<'SLURM' | awk '{print $NF}'
#!/bin/bash
#SBATCH --job-name=tm3d_eps
#SBATCH --output=eps_sweep_3d.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --partition=express

PROJECT_DIR="${HOME}/Projects/twin_mortar/IGAros"

echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Date: $(date)"
echo "Results → ${TM_RESULTS_DIR}"

julia --project="${PROJECT_DIR}" -e 'import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()'
echo ">>> Precompile done: $(date)"

julia -t ${SLURM_CPUS_PER_TASK:-8} --project="${PROJECT_DIR}" "${PROJECT_DIR}/examples/run_eps_sweep_3d.jl"
echo ">>> Finished: $(date)"
SLURM
)
echo "Submitted eps_sweep:          job ${jid2}"

# ── Job 3: NQUAD sweep ───────────────────────────────────────────────────────
jid3=$(sbatch --export=ALL <<'SLURM' | awk '{print $NF}'
#!/bin/bash
#SBATCH --job-name=tm3d_nquad
#SBATCH --output=nquad_sweep_3d.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=01:30:00
#SBATCH --partition=express

PROJECT_DIR="${HOME}/Projects/twin_mortar/IGAros"

echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Date: $(date)"
echo "Results → ${TM_RESULTS_DIR}"

julia --project="${PROJECT_DIR}" -e 'import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()'
echo ">>> Precompile done: $(date)"

julia -t ${SLURM_CPUS_PER_TASK:-8} --project="${PROJECT_DIR}" "${PROJECT_DIR}/examples/run_nquad_sweep_3d.jl"
echo ">>> Finished: $(date)"
SLURM
)
echo "Submitted nquad_sweep:        job ${jid3}"

echo ""
echo "All jobs submitted:"
echo "  factorial_moments:  ${jid1}"
echo "  eps_sweep:          ${jid2}"
echo "  nquad_sweep:        ${jid3}"
echo ""
echo "Results directory:    ${RESULTS_DIR}"
