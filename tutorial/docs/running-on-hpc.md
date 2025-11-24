# Running on HPC

## ABI HPC Standard

If running on the ABI HPC, you can use a pre-installed (with all neccesary libraries) OpenCOR pythonshell. To do this, you need to clone the circulatory\_autogen repo into a directory of your choice, then change the opencor\_pythonshell\_path in opencor\_pythonshell\_path.sh to...

`opencor_pythonshell_path=/hpc/farg967/OpenCOR-0-8-1-Linux/pythonshell`

Following that to run in parallel you need to load mpich. Do the following from the {project_dir}/user_run_files dir:

`. load_mpi.sh`

Then you should be able to run as normal from the user\_run\_files dir

## ABI HPC Extra

If you need to install specific python libraries that aren't installed in the above OpenCOR python version, then you 
will need to install a separate OpenCOR and install the libraries. See getting-started for how to install python libraries, 
the process should be the same on the HPC.

!!! warning
    Before installing mpi4py in your separate OpenCOR, make sure you load mpi with
    
    `module load mpi/mpich-x86_64 && echo "succesfully loaded mpi/mpich-x86_64"`
