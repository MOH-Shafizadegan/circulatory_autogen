#!/bin/bash
source opencor_pythonshell_path.sh

echo "Running identifiability analysis with $1 processor"

mpiexec -n "$1" "${opencor_pythonshell_path}" ../src/scripts/profile_likelihood_run_script.py