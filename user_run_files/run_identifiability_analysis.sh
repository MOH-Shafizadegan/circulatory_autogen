#!/bin/bash
source opencor_pythonshell_path.sh

echo "Running identifiability analysis with 1 processor"

${opencor_pythonshell_path} ../src/scripts/identifiability_run_script.py
