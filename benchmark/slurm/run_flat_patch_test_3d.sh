#!/bin/bash
#SBATCH --job-name=tm_flat3d
#SBATCH --output=flat_patch_test_3d.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --partition=express

PROJECT_DIR="${HOME}/Projects/twin_mortar/IGAros"

echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Date: $(date)"
echo "Results → ${PROJECT_DIR}/../../results/flat_patch_test_3d/"

julia --project="${PROJECT_DIR}" -e '
    import Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()
'
echo ">>> Precompile done: $(date)"

julia -t ${SLURM_CPUS_PER_TASK:-8} \
    --project="${PROJECT_DIR}" \
    "${PROJECT_DIR}/examples/run_flat_patchtest_3d.jl"

echo ">>> Finished: $(date)"
