#!/bin/bash

# Source the .bashrc file
source ~/.bashrc

# Activate the conda environment
conda activate compneuro

# Change to the specified directory
cd ~/successor_examples

# Stash any changes in the git repository
git stash

# Pull the latest changes from the remote repository
git pull

# Submit the job multiple times using sbatch
sbatch rats.batch

# Monitor the job queue
watch -n 1 squeue