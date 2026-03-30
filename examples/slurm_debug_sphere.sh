#!/bin/bash
#SBATCH --job-name=sphere_debug
#SBATCH --partition=express
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 0-4:00:00
#SBATCH -o sphere_debug_%j.out
#SBATCH -e sphere_debug_%j.err

cd ~/Projects/twin_mortar
~/julia/bin/julia --project=IGAros IGAros/examples/debug_sphere_level2.jl
