import sys
import os
from mpi4py import MPI
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
import traceback
import yaml
from parsers.PrimitiveParsers import YamlFileParser
from mpi4py import MPI
from param_id.paramID import CVS0DParamID

def run_SA(inp_data_dict=None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == 0:
        print(f'Running sensitivity analysis with {num_procs} MPI rank(s)')

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)


    # SA_agent = SensitivityAnalysis(model_path=model_path, model_type=model_type, file_name_prefix=file_name_prefix,
    #                                DEBUG=DEBUG, model_out_names=model_out_names, solver_info=solver_info, dt=dt, 
    #                                ga_options=optimiser_options, param_id_obs_path=param_id_obs_path, params_for_id_path=params_for_id_path)
    SA_agent = SensitivityAnalysis.from_dict(inp_data_dict)
    SA_agent.run_sensitivity_analysis()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_SA()
        MPI.Finalize()
    except:
        print(traceback.format_exc())
        comm.Abort()
        MPI.Finalize()