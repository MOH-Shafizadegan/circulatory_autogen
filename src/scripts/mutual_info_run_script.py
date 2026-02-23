import sys  
import os  
from mpi4py import MPI  
root_dir = os.path.join(os.path.dirname(__file__), '../..')  
sys.path.append(os.path.join(root_dir, 'src'))  
  
from mutual_information.mutualInfo import mutualInfo  
import traceback  
import yaml  
from parsers.PrimitiveParsers import YamlFileParser  
  
def run_mutual_info_analysis(inp_data_dict=None):  
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()  
    num_procs = comm.Get_size()  
    if rank == 0:  
        print(f'Running mutual information analysis with {num_procs} MPI rank(s)')  
  
    yaml_parser = YamlFileParser()  
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)  
  
    # Extract configuration  
    model_path = inp_data_dict['model_path']  
    model_out_names = inp_data_dict.get('model_out_names', [])  
    solver_info = inp_data_dict['solver_info']  
    dt = inp_data_dict['dt']  
    param_id_obs_path = inp_data_dict['param_id_obs_path']  
    params_for_id_path = inp_data_dict['params_for_id_path']  
    mi_options = inp_data_dict.get('mi_options', {})  
      
    # Set defaults  
    mi_options.setdefault('num_samples', 128)  
    mi_options.setdefault('n_shuffles', 100)  
    mi_options.setdefault('output_dir', os.path.join(inp_data_dict['param_id_output_dir'], 'mutual_info_results'))  
  
    # Create and run MI analysis  
    mi_analyzer = mutualInfo(  
        model_path=model_path,  
        model_out_names=model_out_names,  
        solver_info=solver_info,  
        SA_info=mi_options,  
        dt=dt,  
        sa_output_dir=mi_options['output_dir'],  
        param_id_path=param_id_obs_path,  
        params_for_id_path=params_for_id_path,  
        use_MPI=True,  
        verbose=False  
    )  
      
    mi_analyzer.run()  
  
    if rank == 0:  
        print(f"Mutual information analysis complete. Results saved to {mi_options['output_dir']}")  
  
if __name__ == '__main__':  
    comm = MPI.COMM_WORLD  
    try:  
        run_mutual_info_analysis()  
        MPI.Finalize()  
    except:  
        print(traceback.format_exc())  
        comm.Abort()  
        MPI.Finalize()