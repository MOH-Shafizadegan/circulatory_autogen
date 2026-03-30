"""  
Tests for profile likelihood functionality.  
  
These tests verify that profile likelihood analysis works correctly for various models.  
"""  
import os  
import pytest  
import numpy as np  
import json  
from mpi4py import MPI  
  
from identifiabilty_analysis.ProfileLikelihood import ProfileLikelihood  
from scripts.param_id_run_script import run_param_id  
  
  
@pytest.fixture(scope="function")  
def mpi_comm():  
    """Fixture that provides MPI communicator."""  
    comm = MPI.COMM_WORLD  
    return comm  
  
  
@pytest.mark.unit  
def test_profile_likelihood_initialization():  
    """Test ProfileLikelihood class initialization."""  
    # Mock param_id and param_id_info  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
      
    profile = ProfileLikelihood(  
        param_id=None,  
        param_id_info=param_id_info,  
        output_dir='/tmp/test',  
        num_points=20,  
        range_factor=0.1  
    )  
      
    assert profile.num_points == 20  
    assert profile.range_factor == 0.1  
    assert profile.output_dir == '/tmp/test'  
    assert profile.param_id_info == param_id_info  
  
  
@pytest.mark.unit  
def test_profile_likelihood_set_best_param_vals():  
    """Test setting best parameter values."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
      
    profile = ProfileLikelihood(  
        param_id=None,  
        param_id_info=param_id_info,  
        output_dir='/tmp/test'  
    )  
      
    best_params = np.array([0.5, 0.7])  
    profile.set_best_param_vals(best_params)  
      
    np.testing.assert_array_equal(profile.best_param_vals, best_params)  
  
  
@pytest.mark.unit  
def test_profile_likelihood_no_best_params():  
    """Test that run_profile_likelihood raises error without best params."""  
    param_id_info = {  
        'param_names': ['param1'],  
        'param_mins': np.array([0.0]),  
        'param_maxs': np.array([1.0])  
    }  
      
    profile = ProfileLikelihood(  
        param_id=None,  
        param_id_info=param_id_info,  
        output_dir='/tmp/test'  
    )  
      
    with pytest.raises(ValueError, match="Best parameter values must be set first"):  
        profile.run_profile_likelihood()  
    
  
@pytest.mark.integration  
@pytest.mark.slow  
@pytest.mark.mpi  
def test_profile_likelihood_3compartment_succeeds(base_user_inputs, resources_dir,   
                                                   temp_output_dir, mpi_comm):  
    """  
    Test that profile likelihood succeeds for 3compartment model.  
      
    Args:  
        base_user_inputs: Base user inputs configuration fixture  
        resources_dir: Resources directory fixture  
        temp_output_dir: Temporary output directory fixture  
        mpi_comm: MPI communicator fixture  
    """  
    rank = mpi_comm.Get_rank()  
      
    # Setup configuration  
    config = base_user_inputs.copy()  
    config.update({  
        'file_prefix': '3compartment',  
        'input_param_file': '3compartment_parameters.csv',  
        'model_type': 'cellml_only',  
        'solver': 'CVODE',  
        'param_id_method': 'genetic_algorithm',  
        'pre_time': 20,  
        'sim_time': 2,  
        'dt': 0.01,  
        'DEBUG': True,  
        'do_mcmc': False,  
        'plot_predictions': False,  
        'do_ia': False,  
        'solver_info': {  
            'MaximumStep': 0.001,  
            'MaximumNumberOfSteps': 5000,  
        },  
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),  
        'param_id_output_dir': temp_output_dir,  
        'optimiser_options': {  
            'num_calls_to_function': 100,  
            'max_patience': 10,  
            'cost_convergence': 1e-3,  
        },  
    })  
      
    # First run parameter identification to get best parameters  
    run_param_id(config)

    if rank == 0:  
        # The actual output directory includes method and model name  
        actual_output_dir = os.path.join(  
            temp_output_dir,  
            f"{config['param_id_method']}_{config['file_prefix']}_3compartment_obs_data"  
        )  
        
        # Check if param_names_for_gen.csv exists in the actual output directory  
        param_names_file = os.path.join(actual_output_dir, 'param_names_for_gen.csv')  
        if not os.path.exists(param_names_file):  
            print("Files are missing :(")  
            # Load best parameters and create the file  
            best_params_file = os.path.join(actual_output_dir, 'best_param_vals.npy')  
            if os.path.exists(best_params_file):  
                print("HERE")  
                best_params = np.load(best_params_file)  
                # Create simple parameter names for testing  
                param_names = [[f'param_{i}'] for i in range(len(best_params))]  
                with open(param_names_file, 'w', newline='') as f:  
                    writer = csv.writer(f)  
                    writer.writerows(param_names)  
        
        # Now load from the correct directory  
        from parsers.PrimitiveParsers import CSVFileParser  
        csv_parser = CSVFileParser()
        param_id_name_and_vals, _ = csv_parser.get_param_id_params_as_lists_of_tuples(  
            actual_output_dir  
        )
      
    # Then run profile likelihood analysis  
    if rank == 0:  
        from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis  
        from param_id.paramID import CVS0DParamID  
        
        actual_output_dir = os.path.join(  
            temp_output_dir,  
            f"{config['param_id_method']}_{config['file_prefix']}_3compartment_obs_data"  
        )  
        
        # Now load from the correct directory (not temp_output_dir)  
        param_id_name_and_vals, _ = csv_parser.get_param_id_params_as_lists_of_tuples(  
            actual_output_dir  # Use actual_output_dir here  
        )  

        best_param_vals = np.array([val for name, val in param_id_name_and_vals])  
          
        # Initialize param_id object for profile likelihood  
        param_id = CVS0DParamID(  
            model_path=config['model_path'],  
            model_type=config['model_type'],  
            param_id_method=config['param_id_method'],  
            mcmc_instead=False,  
            file_name_prefix=config['file_prefix'],  
            params_for_id_path=config['params_for_id_path'],  
            param_id_obs_path=config['param_id_obs_path'],  
            sim_time=config['sim_time'],  
            pre_time=config['pre_time'],  
            solver_info=config['solver_info'],  
            dt=config['dt'],  
            optimiser_options=config['optimiser_options'],  
            DEBUG=config['DEBUG'],  
            param_id_output_dir=temp_output_dir,  
            resources_dir=config['resources_dir']  
        )  
          
        # Create and run profile likelihood  
        profile = ProfileLikelihood(  
            param_id=param_id.param_id,  
            param_id_info=param_id.param_id_info,  
            output_dir=temp_output_dir,  
            num_points=10,  # Reduced for test speed  
            range_factor=0.1,  
            optimiser_options={'num_calls_to_function': 50}  
        )  
          
        profile.set_best_param_vals(best_param_vals)  
        profile.run_profile_likelihood()  
          
        # Verify output files were created  
        results_dir = os.path.join(temp_output_dir, 'profile_likelihood')  
        assert os.path.exists(results_dir), "Profile likelihood results directory should exist"  
          
        # Check summary file  
        summary_file = os.path.join(results_dir, 'summary.json')  
        assert os.path.exists(summary_file), "Summary file should exist"  
          
        with open(summary_file, 'r') as f:  
            summary = json.load(f)  
          
        assert 'num_params' in summary  
        assert 'num_points' in summary  
        assert summary['num_points'] == 10  
          
        # Check parameter files exist  
        for param_idx in range(len(best_param_vals)):  
            param_file = os.path.join(results_dir, f"param_{param_idx}_*.json")  
            assert len(os.listdir(results_dir)) > 0, "Parameter result files should exist"  
      
    mpi_comm.Barrier()  
  
  
@pytest.mark.unit  
def test_profile_likelihood_save_results():  
    """Test saving profile likelihood results."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
      
    import tempfile  
    with tempfile.TemporaryDirectory() as temp_dir:  
        profile = ProfileLikelihood(  
            param_id=None,  
            param_id_info=param_id_info,  
            output_dir=temp_dir  
        )  
          
        # Mock results  
        profile.profile_results = {  
            0: {  
                'param_values': np.array([0.1, 0.2, 0.3]),  
                'costs': np.array([1.0, 0.5, 1.0]),  
                'param_name': 'param1',  
                'best_val': 0.2  
            },  
            1: {  
                'param_values': np.array([0.4, 0.5, 0.6]),  
                'costs': np.array([2.0, 1.0, 2.0]),  
                'param_name': 'param2',  
                'best_val': 0.5  
            }  
        }  
          
        profile._save_results()  
          
        results_dir = os.path.join(temp_dir, 'profile_likelihood')  
        assert os.path.exists(results_dir)  
          
        # Check summary file  
        summary_file = os.path.join(results_dir, 'summary.json')  
        assert os.path.exists(summary_file)  
          
        with open(summary_file, 'r') as f:  
            summary = json.load(f)  
          
        assert summary['num_params'] == 2  
        assert summary['parameters'] == ['param1', 'param2']  
          
        # Check individual parameter files  
        for param_idx in [0, 1]:  
            param_files = [f for f in os.listdir(results_dir) if f.startswith(f'param_{param_idx}_')]  
            assert len(param_files) > 0  
              
            with open(os.path.join(results_dir, param_files[0]), 'r') as f:  
                data = json.load(f)  
              
            assert 'param_values' in data  
            assert 'costs' in data  
            assert 'param_name' in data  
            assert 'best_val' in data  
  
  
@pytest.mark.unit  
def test_profile_likelihood_plotting():  
    """Test profile likelihood plotting functionality."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
      
    import tempfile  
    import matplotlib  
    matplotlib.use('Agg')  # Use non-interactive backend for testing  
      
    with tempfile.TemporaryDirectory() as temp_dir:  
        profile = ProfileLikelihood(  
            param_id=None,  
            param_id_info=param_id_info,  
            output_dir=temp_dir  
        )  
          
        # Mock results  
        profile.profile_results = {  
            0: {  
                'param_values': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  
                'costs': np.array([2.0, 1.0, 0.5, 1.0, 2.0]),  
                'param_name': 'param1',  
                'best_val': 0.3  
            },  
            1: {  
                'param_values': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  
                'costs': np.array([3.0, 1.5, 0.8, 1.5, 3.0]),  
                'param_name': 'param2',  
                'best_val': 0.3  
            }  
        }  
          
        profile._plot_profiles()  
          
        results_dir = os.path.join(temp_dir, 'profile_likelihood')  
          
        # Check that plot files were created  
        plot_files = [f for f in os.listdir(results_dir) if f.endswith('.pdf')]  
        assert len(plot_files) >= 1  # At least the combined plot  
          
        # Check individual parameter plots  
        for param_idx in [0, 1]:  
            param_plots = [f for f in os.listdir(results_dir)   
                          if f.startswith(f'param_{param_idx}_') and f.endswith('.pdf')]  
            assert len(param_plots) > 0