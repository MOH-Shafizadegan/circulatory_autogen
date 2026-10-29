"""  
Tests for profile likelihood functionality.  
  
These tests verify that profile likelihood analysis works correctly for various models.  
"""  
import os  
import json  
import tempfile  
  
import pytest  
import numpy as np  
from mpi4py import MPI  
  
from libcuflynx.identifiabilty_analysis.ProfileLikelihood import ProfileLikelihood  
from libcuflynx.scripts.param_id_run_script import run_param_id  
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture  
  
  
@pytest.fixture(scope="function")  
def mpi_comm():  
    """Fixture that provides MPI communicator."""  
    comm = MPI.COMM_WORLD  
    return comm  
  
  
def _ensure_cellml_model_generated(config, mpi_comm):  
    """Ensure generated CellML exists before run_param_id (mirrors test_param_id.py)."""  
    if config.get("model_type") != "cellml":  
        return  
    rank = mpi_comm.Get_rank()  
    if rank == 0:  
        success = generate_with_new_architecture(False, config)  
        prefix = config.get("file_prefix", "<unknown>")  
        assert success, f"CellML autogeneration failed for {prefix}"  
    mpi_comm.Barrier()  
  
  
@pytest.mark.unit  
def test_profile_likelihood_initialization():  
    """Test ProfileLikelihood class initialization."""  
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
        profile._run()  
  
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_profile_likelihood_3compartment_succeeds(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """Calibrate, then profile both parameters through ``do_ia`` on every rank."""
    rank = mpi_comm.Get_rank()

    # Two-parameter params_for_id, written once on rank 0.
    params_for_id_path = os.path.join(
        temp_output_dir, "3compartment_profile_likelihood_params_for_id.json"
    )
    if rank == 0:
        with open(params_for_id_path, "w") as f:
            json.dump({"version": 1, "params": [
                {"targets": ["aortic_root/C"], "min": 1e-9, "max": 5e-8,
                 "param_type": "const", "name_for_plotting": "C_{ao}"},
                {"targets": ["global/q_lv_init"], "min": 200e-6, "max": 1500e-6,
                 "param_type": "const", "name_for_plotting": "q_{sbv}"},
            ]}, f)
    mpi_comm.Barrier()

    num_points = 3
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.json',
        'params_for_id_file': params_for_id_path,
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'plot_predictions': False,
        'do_uq': False,
        'do_ia': True,
        # Bounded: 2 parameters x 3 points, each a short GA re-fit.
        'ia_options': {'method': 'profile_likelihood', 'num_points': num_points,
                       'range_factor': 0.1},
        'solver_info': {'MaximumStep': 0.001},
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 60, 'max_patience': 500,
                                    'cost_type': 'gaussian_MLE'},
    })

    _ensure_cellml_model_generated(config, mpi_comm)
    result = run_param_id(config)

    # The profiles leave the calibration as they found it.
    inner = result['param_id'].param_id
    assert [names for names in inner.param_id_info['param_names']] == \
        [['aortic_root/C'], ['global/q_lv_init']]
    assert inner.num_params == 2

    if rank == 0:
        results_dir = os.path.join(temp_output_dir, 'profile_likelihood')
        with open(os.path.join(results_dir, 'summary.json')) as f:
            summary = json.load(f)
        assert summary['num_params'] == 2
        assert summary['num_points'] == num_points
        mins = [1e-9, 200e-6]
        maxs = [5e-8, 1500e-6]
        for param_idx in range(2):
            json_files = [f for f in os.listdir(results_dir)
                          if f.startswith(f'param_{param_idx}_') and f.endswith('.json')]
            assert len(json_files) == 1
            with open(os.path.join(results_dir, json_files[0])) as f:
                profile = json.load(f)
            values = np.array(profile['param_values'])
            assert len(values) == num_points
            assert np.all(values >= mins[param_idx]) and np.all(values <= maxs[param_idx])
            assert np.all(np.isfinite(profile['costs']))

    mpi_comm.Barrier()

@pytest.mark.unit  
def test_profile_likelihood_save_results():  
    """Test saving profile likelihood results."""  
    param_id_info = {  
        'param_names': ['param1', 'param2'],  
        'param_mins': np.array([0.0, 0.0]),  
        'param_maxs': np.array([1.0, 1.0])  
    }  
  
    with tempfile.TemporaryDirectory() as temp_dir:  
        profile = ProfileLikelihood(  
            param_id=None,  
            param_id_info=param_id_info,  
            output_dir=temp_dir  
        )  
  
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
  
        summary_file = os.path.join(results_dir, 'summary.json')  
        assert os.path.exists(summary_file)  
  
        with open(summary_file, 'r') as f:  
            summary = json.load(f)  
  
        assert summary['num_params'] == 2  
        assert summary['parameters'] == ['param1', 'param2']  
  
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