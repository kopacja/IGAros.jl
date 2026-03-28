#!/bin/bash
#
# SLURM scaling benchmark for IGAros stiffness assembly
# Submits one job per thread count: 1, 2, 4, 8, 16, 32, 48, 96
#
# Usage:  cd IGAros/benchmark && bash bench_assembly.sh
#
# Prerequisites:
#   - Julia installed / available via module (adjust JULIA_BIN below)
#   - IGAros project dependencies installed:
#       julia --project=.. -e 'using Pkg; Pkg.instantiate()'
#
# Results land in bench_assembly_<N>t.out

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Adjust these to match your cluster setup ─────────────────────────────────
JULIA_BIN="julia"                         # or: module load julia && julia
PARTITION="express"                       # 6h limit, all nodes available
TIME="01:00:00"                           # 1 hour should be plenty
MEM="32gb"                                # peak ~4 GB for largest case
# ─────────────────────────────────────────────────────────────────────────────

for NT in 1 2 4 8 16 32 48 96; do
    JOBNAME="iga_bench_${NT}t"
    OUTFILE="${SCRIPT_DIR}/bench_assembly_${NT}t.out"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${OUTFILE}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NT}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --exclusive

echo "Host: \$(hostname)"
echo "CPUs allocated: \${SLURM_CPUS_PER_TASK}"
echo "Date: \$(date)"
echo ""

${JULIA_BIN} -t ${NT} --project=${SCRIPT_DIR}/.. ${SCRIPT_DIR}/bench_assembly.jl
EOF

    echo "Submitted ${JOBNAME} (${NT} threads) → ${OUTFILE}"
done
