"""Tests for profile likelihood (``libcuflynx.identifiabilty_analysis.ProfileLikelihood``)."""

import json
import os
import shutil
import tempfile

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from _params_for_id import write_params_for_id
from libcuflynx.identifiabilty_analysis.ProfileLikelihood import (
    NLL_THRESHOLD_95,
    ProfileLikelihood,
    profile_likelihood_options,
)
from libcuflynx.param_id.paramID import ParamID
from libcuflynx.parsers.PrimitiveParsers import analysis_options
from libcuflynx.scripts.param_id_run_script import run_param_id
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities.mpi_utils import get_MPI
from libcuflynx.utilities.utility_funcs import Normalise_class


@pytest.fixture(scope="function")
def mpi_comm():
    """Fixture that provides MPI communicator."""
    # mpi4py is optional: get_MPI is a one-rank stand-in without it.
    comm = get_MPI().COMM_WORLD
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


# ---- a ParamID with no model behind it -----------------------------------------------------


class _RecordingSimHelper:
    """Records every set_param_vals call as {name: value}."""

    def __init__(self):
        self.values = {}
        self.calls = []

    def set_param_vals(self, param_names, param_vals, change_states=True):
        self.calls.append((param_names, param_vals))
        for names, vals in zip(param_names, param_vals):
            names = names if isinstance(names, list) else [names]
            vals = (
                list(vals) if isinstance(vals, (list, tuple, np.ndarray)) else [vals] * len(names)
            )
            self.values.update(zip(names, vals))


def _stub_param_id(tmp_dir):
    """A real ParamID (its remove_params_by_idx and state) with a recording sim helper.

    Entries: 0 a group [A, B], 1 a scale modifier on [C, D] (baselines 2, 3), 2 a plain E.
    """
    pid = object.__new__(ParamID)
    mins, maxs = np.array([0.0, 0.5, 0.0]), np.array([10.0, 4.0, 10.0])
    pid.param_id_info = {
        "param_names": [["g/A", "g/B"], ["m/C", "m/D"], ["p/E"]],
        "param_mins": mins.copy(),
        "param_maxs": maxs.copy(),
        "param_prior_types": np.array(["uniform"] * 3),
        "param_names_for_plotting": np.array(["A", "mod", "E"]),
        "param_entry_names": ["grp", "mod", "E"],
        "param_labels": ["g/A+g/B", "mod", "p/E"],
        "modifiers": [
            {
                "index": 1,
                "name": "mod",
                "modifier": "scale",
                "targets": ["m/C", "m/D"],
                "baselines": [2.0, 3.0],
                "inputs": {},
                "resolved_inputs": {},
                "affine": {"a": [2.0, 3.0], "b": [0.0, 0.0]},
            }
        ],
        "modifier_funcs_external_path": None,
    }
    pid.num_params = 3
    pid.param_norm_obj = Normalise_class(mins, maxs)
    pid.param_init = None
    pid.best_param_vals = np.array([5.0, 1.0, 5.0])
    pid.best_cost = 0.25
    pid.optimiser_options = {"num_calls_to_function": 1000, "cost_type": "gaussian_MLE"}
    pid.output_dir = str(tmp_dir)
    pid.sim_helper = _RecordingSimHelper()
    pid.emulates_features = False
    pid.get_cost_from_params = lambda vals: float(np.sum(np.asarray(vals) ** 2))
    return pid


def _profile_for(pid):
    profile = ProfileLikelihood(
        param_id=pid,
        param_id_info=pid.param_id_info,
        output_dir=pid.output_dir,
        num_points=3,
        range_factor=0.1,
        optimiser_options={"num_calls_to_function": 40},
    )
    profile.set_best_param_vals(pid.best_param_vals)
    return profile


@pytest.mark.unit
def test_remove_params_by_idx_reindexes_modifiers_and_per_entry_lists(tmp_path):
    pid = _stub_param_id(tmp_path)
    pid.remove_params_by_idx([0])
    info = pid.param_id_info
    assert info["param_names"] == [["m/C", "m/D"], ["p/E"]]
    assert list(info["param_names_for_plotting"]) == ["mod", "E"]
    assert info["param_entry_names"] == ["mod", "E"]
    assert info["param_labels"] == ["mod", "p/E"]
    assert [m["index"] for m in info["modifiers"]] == [0]

    pid = _stub_param_id(tmp_path)
    pid.remove_params_by_idx([1])
    assert pid.param_id_info["modifiers"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "idx, value, expected",
    [
        (0, 7.0, {"g/A": 7.0, "g/B": 7.0}),  # a group is held as a whole
        (1, 2.0, {"m/C": 4.0, "m/D": 6.0}),  # a modifier is held through its function
    ],
)
def test_a_profile_point_holds_the_whole_entry_and_recalibrates_the_rest(
    tmp_path, idx, value, expected
):
    pid = _stub_param_id(tmp_path)
    original_info = pid.param_id_info
    original_options = pid.optimiser_options
    seen = {}

    def fake_run(param_init=None):
        info = pid.param_id_info
        seen["names"] = list(info["param_names"])
        seen["modifier_indices"] = [m["index"] for m in info["modifiers"]]
        seen["model"] = dict(pid.sim_helper.values)
        seen["options"] = dict(pid.optimiser_options)
        seen["param_init"] = param_init
        pid.best_param_vals = np.array([1.5, 2.5])
        pid.best_cost = 9.0

    pid.run = fake_run
    cost, optimum = _profile_for(pid)._optimize_with_fixed_parameter(idx, value)

    remaining = [n for i, n in enumerate(original_info["param_names"]) if i != idx]
    assert seen["names"] == remaining
    for name, held in expected.items():
        assert seen["model"][name] == pytest.approx(held)
    # The modifier (entry 1) keeps pointing at itself once entry 0 is removed.
    assert seen["modifier_indices"] == ([0] if idx == 0 else [])
    # ia_options.optimiser_options are merged over the calibration's.
    assert seen["options"] == {"num_calls_to_function": 40, "cost_type": "gaussian_MLE"}
    np.testing.assert_allclose(seen["param_init"], np.delete([5.0, 1.0, 5.0], idx))

    np.testing.assert_allclose(optimum, [1.5, 2.5])
    assert cost == pytest.approx(1.5**2 + 2.5**2)
    # Restored, and the caller's dict was never reduced.
    assert pid.param_id_info is original_info
    assert original_info["param_names"] == [["g/A", "g/B"], ["m/C", "m/D"], ["p/E"]]
    assert [m["index"] for m in original_info["modifiers"]] == [1]
    assert pid.optimiser_options is original_options
    assert original_options == {"num_calls_to_function": 1000, "cost_type": "gaussian_MLE"}
    assert pid.num_params == 3 and pid.best_cost == 0.25
    np.testing.assert_allclose(pid.best_param_vals, [5.0, 1.0, 5.0])
    assert pid.output_dir == str(tmp_path)


@pytest.mark.unit
def test_a_profile_point_runs_on_a_rank_without_an_output_dir(tmp_path):
    """Ranks other than 0 hold output_dir None; every rank must reach the collective run()."""
    pid = _stub_param_id(tmp_path)
    pid.output_dir = None
    ran = []

    def fake_run(param_init=None):
        ran.append(pid.output_dir)
        pid.best_param_vals = np.array([1.0, 1.0])

    pid.run = fake_run
    _profile_for(pid)._optimize_with_fixed_parameter(0, 7.0)
    assert ran == [None] and pid.output_dir is None


@pytest.mark.unit
def test_state_is_restored_when_a_profile_point_fails(tmp_path):
    pid = _stub_param_id(tmp_path)
    original_info = pid.param_id_info
    original_norm = pid.param_norm_obj

    def failing_run(param_init=None):
        pid.best_param_vals = np.array([0.0, 0.0])
        raise RuntimeError("solver blew up")

    pid.run = failing_run
    with pytest.raises(RuntimeError, match="solver blew up"):
        _profile_for(pid)._optimize_with_fixed_parameter(0, 7.0)

    assert pid.param_id_info is original_info
    assert len(original_info["param_names"]) == 3
    assert pid.num_params == 3
    assert pid.param_norm_obj is original_norm
    assert pid.optimiser_options == {"num_calls_to_function": 1000, "cost_type": "gaussian_MLE"}
    assert pid.output_dir == str(tmp_path)
    np.testing.assert_allclose(pid.best_param_vals, [5.0, 1.0, 5.0])
    # The held entry goes back to its best-fit value.
    assert pid.sim_helper.values["g/A"] == pid.sim_helper.values["g/B"] == 5.0


@pytest.mark.unit
def test_each_profile_point_warm_starts_from_its_neighbours_optimum(tmp_path):
    """The sweep runs outwards from the best fit, each fit starting at the last one's optimum."""
    pid = _stub_param_id(tmp_path)
    starts = []

    def fake_run(param_init=None):
        starts.append(None if param_init is None else np.array(param_init))
        held = pid.sim_helper.values["p/E"]
        pid.best_param_vals = np.array([held, held + 100.0])  # a recognisable optimum per point

    pid.run = fake_run
    profile = _profile_for(pid)
    profile.num_points = 5
    result = profile._profile_single_parameter(2)

    values = result["param_values"]
    centre = int(np.argmin(np.abs(values - 5.0)))
    assert centre == 2
    order = [2, 3, 4, 1, 0]
    expected = {2: [5.0, 1.0], 3: [values[2], values[2] + 100], 4: [values[3], values[3] + 100]}
    expected[1] = [values[2], values[2] + 100]
    expected[0] = [values[1], values[1] + 100]
    assert len(starts) == 5
    for point, start in zip(order, starts):
        np.testing.assert_allclose(start, expected[point])


# ---- the 95% threshold -----------------------------------------------------------------------


@pytest.mark.unit
def test_the_95_percent_interval_uses_the_negative_log_likelihood_scale():
    """cost = -ln L / N, so the bound is a rise of 1.92 / N in the cost, not 3.84."""
    sigma, centre, n_obs = 0.1, 0.5, 10.0
    values = np.linspace(0.0, 1.0, 4001)
    nll = (values - centre) ** 2 / (2 * sigma**2)
    pid = type("P", (), {"_lnlikelihood_denorm_factor": n_obs})()
    profile = ProfileLikelihood(pid, None, "/unused")
    ci = profile.confidence_interval_95(values, nll / n_obs, best_val=centre)
    half_width = sigma * np.sqrt(2 * NLL_THRESHOLD_95)  # 1.96 sigma
    assert NLL_THRESHOLD_95 == pytest.approx(3.841459 / 2, rel=1e-6)
    assert ci["lower"] == pytest.approx(centre - half_width, abs=1e-3)
    assert ci["upper"] == pytest.approx(centre + half_width, abs=1e-3)
    assert ci["cost_threshold"] == pytest.approx(NLL_THRESHOLD_95 / n_obs)

    # A sweep that never rises far enough leaves that side open.
    flat = profile.confidence_interval_95(values, np.where(values < centre, 0.0, nll / n_obs))
    assert flat["lower"] is None and flat["upper"] is not None


# ---- options -------------------------------------------------------------------------------


@pytest.mark.unit
def test_profile_likelihood_options_default_from_the_schema_and_are_checked():
    schema = {o["name"]: o["default"] for o in analysis_options("identifiability_analysis")}
    options = profile_likelihood_options({"method": "profile_likelihood"})
    assert options == {
        "num_points": schema["num_points"],
        "range_factor": schema["range_factor"],
        "optimiser_options": {},
    }
    assert profile_likelihood_options(
        {"num_points": 5, "range_factor": 0.5, "optimiser_options": {"seed": 1}}
    ) == {"num_points": 5, "range_factor": 0.5, "optimiser_options": {"seed": 1}}
    for bad in (
        {"num_points": 1},
        {"num_points": 2.5},
        {"num_points": True},
        {"range_factor": 0},
        {"range_factor": 1.5},
        {"range_factor": "0.2"},
        {"optimiser_options": [1]},
    ):
        with pytest.raises(ValueError):
            profile_likelihood_options(bad)


# ---- construction, saving and plotting -----------------------------------------------------


@pytest.mark.unit
def test_profile_likelihood_initialization():
    param_id_info = {
        "param_names": ["param1", "param2"],
        "param_mins": np.array([0.0, 0.0]),
        "param_maxs": np.array([1.0, 1.0]),
    }
    profile = ProfileLikelihood(
        param_id=None,
        param_id_info=param_id_info,
        output_dir="/tmp/test",
        num_points=20,
        range_factor=0.1,
    )
    assert profile.num_points == 20
    assert profile.range_factor == 0.1
    assert profile.output_dir == "/tmp/test"
    assert profile.param_id_info == param_id_info


@pytest.mark.unit
def test_profile_likelihood_set_best_param_vals():
    profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir="/tmp/test")
    best_params = np.array([0.5, 0.7])
    profile.set_best_param_vals(best_params)
    np.testing.assert_array_equal(profile.best_param_vals, best_params)


@pytest.mark.unit
def test_profile_likelihood_no_best_params():
    profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir="/tmp/test")
    with pytest.raises(ValueError, match="Best parameter values must be set first"):
        profile.run()


def _mock_results(names, costs_1=(2.0, 1.0, 0.5, 1.0, 2.0)):
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    return {
        0: {
            "param_values": values,
            "costs": np.array(costs_1),
            "param_name": names[0],
            "best_val": 0.3,
        },
        1: {
            "param_values": values,
            "costs": np.array([3.0, 1.5, 0.8, 1.5, 3.0]),
            "param_name": names[1],
            "best_val": 0.3,
        },
    }


@pytest.mark.unit
def test_profile_likelihood_save_results():
    with tempfile.TemporaryDirectory() as temp_dir:
        profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir=temp_dir)
        profile.profile_results = _mock_results(["param1", "param2"])
        profile._save_results()

        results_dir = os.path.join(temp_dir, "profile_likelihood")
        with open(os.path.join(results_dir, "summary.json")) as f:
            summary = json.load(f)
        assert summary["num_params"] == 2
        assert summary["parameters"] == ["param1", "param2"]
        assert summary["nll_threshold_95"] == pytest.approx(NLL_THRESHOLD_95)
        for param_idx in [0, 1]:
            param_files = [
                f for f in os.listdir(results_dir) if f.startswith(f"param_{param_idx}_")
            ]
            with open(os.path.join(results_dir, param_files[0])) as f:
                data = json.load(f)
            assert {"param_values", "costs", "param_name", "best_val", "ci_95"} <= set(data)


@pytest.mark.unit
def test_profile_likelihood_plotting():
    with tempfile.TemporaryDirectory() as temp_dir:
        profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir=temp_dir)
        profile.profile_results = _mock_results(["param1", "param2"])
        profile._plot_profiles()
        results_dir = os.path.join(temp_dir, "profile_likelihood")
        assert "profile_likelihood_plots.pdf" in os.listdir(results_dir)
        for param_idx in [0, 1]:
            assert any(
                f.startswith(f"param_{param_idx}_") and f.endswith(".pdf")
                for f in os.listdir(results_dir)
            )


@pytest.mark.unit
def test_plotting_survives_infinite_costs_and_labels_list_names(monkeypatch):
    """A failed point (cost inf) must not break the y-limits; a list name is not "['...']"."""
    labels = []
    original = matplotlib.axes.Axes.set_xlabel

    def record(self, label, *args, **kwargs):
        labels.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", record)
    with tempfile.TemporaryDirectory() as temp_dir:
        profile = ProfileLikelihood(param_id=None, param_id_info={}, output_dir=temp_dir)
        profile.profile_results = _mock_results(
            [["aortic_root/C"], ["g/A", "g/B"]], costs_1=(np.inf, 1.0, 0.5, 1.0, np.inf)
        )
        profile._plot_profiles()
        files = os.listdir(os.path.join(temp_dir, "profile_likelihood"))
    assert set(labels) == {"aortic_root/C", "g/A+g/B"}
    assert "param_0_aortic_root_C.pdf" in files and "param_1_g_A__g_B.pdf" in files


# ---- end to end ----------------------------------------------------------------------------

_OSCILLATOR_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "funcs_user", "example_model_scipy")
)


@pytest.mark.integration
@pytest.mark.mpi
def test_profile_points_warm_start_through_param_id_run(
    base_user_inputs, temp_output_dir, temp_generated_models_dir, mpi_comm, monkeypatch
):
    """ParamID.run starts each profile point's CMA-ES fit at the neighbouring point's optimum,
    not at the model's parameter values (the scipy oscillator; seconds)."""
    rank = mpi_comm.Get_rank()
    resources_dir = os.path.join(temp_output_dir, "resources")
    if rank == 0:
        os.makedirs(resources_dir, exist_ok=True)
        for name in (
            "oscillator_params_for_id.json",
            "oscillator_parameters.json",
            "oscillator_obs_data.json",
        ):
            shutil.copy(os.path.join(_OSCILLATOR_DIR, name), os.path.join(resources_dir, name))
    mpi_comm.Barrier()

    config = base_user_inputs.copy()
    config.update(
        {
            "file_prefix": "oscillator",
            "input_param_file": "oscillator_parameters.json",
            "model_type": "external_python",
            "solver": "external",
            "solver_info": {"solver": "external", "method": "external"},
            "external_model_path": os.path.join(_OSCILLATOR_DIR, "oscillator_model.py"),
            "resources_dir": resources_dir,
            "param_id_method": "CMA-ES",
            "pre_time": 0.0,
            "sim_time": 10.0,
            "dt": 0.05,
            "DEBUG": True,
            "do_uq": False,
            "do_ad": False,
            "plot_predictions": False,
            "model_out_names": ["oscillator/x"],
            "param_id_obs_path": os.path.join(resources_dir, "oscillator_obs_data.json"),
            "param_id_output_dir": temp_output_dir,
            "generated_models_dir": temp_generated_models_dir,
            "debug_optimiser_options": {
                "num_calls_to_function": 40,
                "cost_type": "gaussian_MLE",
                "seed": 0,
            },
            "do_ia": True,
            "ia_options": {
                "method": "profile_likelihood",
                "num_points": 3,
                "range_factor": 0.1,
                "optimiser_options": {"num_calls_to_function": 12},
            },
        }
    )

    starts, budgets = [], []
    real_run = ParamID.run

    def spy(self, *args, **kwargs):
        real_run(self, *args, **kwargs)
        starts.append(
            (
                self.num_params,
                np.array(self.param_init, dtype=float),
                np.array(self.best_param_vals, dtype=float),
            )
        )
        budgets.append(self.optimiser_options.get("num_calls_to_function"))

    monkeypatch.setattr(ParamID, "run", spy)
    result = run_param_id(config)

    best = np.asarray(result["best_param_vals"], dtype=float)
    calibration, points = starts[0], starts[1:]
    assert calibration[0] == 2 and len(points) == 6
    assert budgets == [40] + [12] * 6
    # Per profiled entry: the first point starts at the best fit, each later one at the
    # optimum of a point already fitted (its neighbour nearer the best fit).
    for p, param_idx in enumerate((0, 1)):
        group = points[3 * p : 3 * p + 3]
        np.testing.assert_allclose(group[0][1], np.delete(best, param_idx))
        np.testing.assert_allclose(group[1][1], group[0][2])
        assert any(np.allclose(group[2][1], g[2]) for g in group[:2])
    assert result["param_id"].param_id.num_params == 2


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_profile_likelihood_3compartment_succeeds(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """Calibrate, then profile both parameters through ``do_ia`` on every rank."""
    rank = mpi_comm.Get_rank()

    params_for_id_path = os.path.join(
        temp_output_dir, "3compartment_profile_likelihood_params_for_id.json"
    )
    if rank == 0:
        write_params_for_id(
            params_for_id_path,
            {
                "version": 1,
                "params": [
                    {
                        "targets": ["aortic_root/C"],
                        "min": 1e-9,
                        "max": 5e-8,
                        "param_type": "const",
                        "name_for_plotting": "C_{ao}",
                    },
                    {
                        "targets": ["global/q_lv_init"],
                        "min": 200e-6,
                        "max": 1500e-6,
                        "param_type": "const",
                        "name_for_plotting": "q_{sbv}",
                    },
                ],
            },
        )
    mpi_comm.Barrier()

    num_points = 3
    config = base_user_inputs.copy()
    config.update(
        {
            "file_prefix": "3compartment",
            "input_param_file": "3compartment_parameters.json",
            "params_for_id_file": params_for_id_path,
            "model_type": "cellml",
            "solver": "CVODE_myokit",
            "param_id_method": "genetic_algorithm",
            "pre_time": 20,
            "sim_time": 2,
            "dt": 0.01,
            "DEBUG": True,
            "plot_predictions": False,
            "do_uq": False,
            "do_ia": True,
            # Bounded: 2 parameters x 3 points, each a short GA re-fit.
            "ia_options": {
                "method": "profile_likelihood",
                "num_points": num_points,
                "range_factor": 0.1,
                "optimiser_options": {"num_calls_to_function": 40},
            },
            "solver_info": {"MaximumStep": 0.001},
            "param_id_obs_path": os.path.join(resources_dir, "3compartment_obs_data.json"),
            "param_id_output_dir": temp_output_dir,
            "generated_models_dir": temp_generated_models_dir,
            "debug_optimiser_options": {
                "num_calls_to_function": 60,
                "max_patience": 500,
                "cost_type": "gaussian_MLE",
            },
        }
    )

    _ensure_cellml_model_generated(config, mpi_comm)
    result = run_param_id(config)

    # The profiles leave the calibration as they found it, inner and outer.
    inner = result["param_id"].param_id
    expected_names = [["aortic_root/C"], ["global/q_lv_init"]]
    assert inner.param_id_info["param_names"] == expected_names
    assert result["param_id"].param_id_info["param_names"] == expected_names
    assert inner.num_params == 2
    assert inner.optimiser_options["num_calls_to_function"] == 60

    if rank == 0:
        results_dir = os.path.join(temp_output_dir, "profile_likelihood")
        with open(os.path.join(results_dir, "summary.json")) as f:
            summary = json.load(f)
        assert summary["num_params"] == 2
        assert summary["num_points"] == num_points
        mins = [1e-9, 200e-6]
        maxs = [5e-8, 1500e-6]
        for param_idx in range(2):
            json_files = [
                f
                for f in os.listdir(results_dir)
                if f.startswith(f"param_{param_idx}_") and f.endswith(".json")
            ]
            assert len(json_files) == 1
            with open(os.path.join(results_dir, json_files[0])) as f:
                profile = json.load(f)
            values = np.array(profile["param_values"])
            assert len(values) == num_points
            assert np.all(values >= mins[param_idx]) and np.all(values <= maxs[param_idx])
            assert np.all(np.isfinite(profile["costs"]))
            # gaussian_MLE is a likelihood, so an interval threshold is recorded.
            assert profile["ci_95"]["cost_threshold"] == pytest.approx(
                min(min(profile["costs"]), profile["best_cost"])
                + NLL_THRESHOLD_95 / profile["nll_per_cost"]
            )

    mpi_comm.Barrier()
