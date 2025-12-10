if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_Laplace_MCMC_tests.sh num_processors'
    exit 1
fi

source opencor_pythonshell_path.sh
mpirun -n "$1" "${opencor_pythonshell_path}" ../src/scripts/run_Laplace_MCMC_tests.py
