from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from tmd.fe.free_energy import HREXCheckpoint, HREXParams, MDParams
from tmd.fe.rbfe import (
    estimate_relative_free_energy_bisection_hrex,
    estimate_relative_free_energy_bisection_hrex_impl,
    estimate_relative_free_energy_bisection_or_hrex,
    run_complex,
    run_solvent,
    run_vacuum,
)


@dataclass
class StubInitialState:
    lamb: float
    x0: object = None
    v0: object = None
    box0: object = None
    barostat: object = None


def make_hrex_md_params() -> MDParams:
    return MDParams(
        n_frames=2,
        n_eq_steps=0,
        steps_per_frame=1,
        seed=2026,
        hrex_params=HREXParams(n_frames_bisection=1),
    )


def make_stub_bisection_output():
    initial_states = [StubInitialState(0.0), StubInitialState(1.0)]
    trajectories = [
        SimpleNamespace(
            frames=[np.zeros((1, 3))],
            boxes=[np.eye(3)],
            final_velocities=np.zeros((1, 3)),
            final_barostat_volume_scale_factor=None,
        )
        for _ in initial_states
    ]
    return [SimpleNamespace(initial_states=initial_states)], trajectories


def make_production_checkpoint(
    completed_frames: int, initial_states_hrex=None, bisection_results=None
) -> HREXCheckpoint:
    return HREXCheckpoint(
        completed_frames=completed_frames,
        hrex=None,
        iterated_u_kln=None,
        replica_idx_by_state_by_iter=[],
        fraction_accepted_by_pair_by_iter=[],
        water_sampler_proposals_by_state_by_iter=[],
        initial_states_hrex=initial_states_hrex,
        bisection_results=bisection_results,
    )


def test_bisection_dispatch_preserves_call_without_checkpoint_arguments():
    md_params = MDParams(n_frames=1, n_eq_steps=0, steps_per_frame=1, seed=2026)
    expected_result = Mock()

    with patch("tmd.fe.rbfe.estimate_relative_free_energy_bisection", return_value=expected_result) as estimate:
        result = estimate_relative_free_energy_bisection_or_hrex("argument", md_params=md_params)

    assert result is expected_result
    estimate.assert_called_once_with("argument", md_params=md_params)


def test_bisection_dispatch_rejects_checkpoint_arguments_without_hrex_params():
    md_params = MDParams(n_frames=1, n_eq_steps=0, steps_per_frame=1, seed=2026)

    with pytest.raises(ValueError, match="require HREX parameters"):
        estimate_relative_free_energy_bisection_or_hrex("argument", md_params=md_params, checkpoint_interval_frames=2)


def test_hrex_dispatch_forwards_checkpoint_arguments():
    md_params = make_hrex_md_params()
    resume_state = Mock(spec=HREXCheckpoint)
    checkpoint_callback = Mock()
    expected_result = Mock()

    with patch(
        "tmd.fe.rbfe.estimate_relative_free_energy_bisection_hrex", return_value=expected_result
    ) as estimate_hrex:
        result = estimate_relative_free_energy_bisection_or_hrex(
            "argument",
            md_params=md_params,
            resume_state=resume_state,
            checkpoint_interval_frames=2,
            checkpoint_callback=checkpoint_callback,
        )

    assert result is expected_result
    estimate_hrex.assert_called_once_with(
        "argument",
        md_params=md_params,
        resume_state=resume_state,
        checkpoint_interval_frames=2,
        checkpoint_callback=checkpoint_callback,
    )


def test_hrex_entry_point_forwards_checkpoint_arguments_to_implementation():
    md_params = make_hrex_md_params()
    resume_state = Mock(spec=HREXCheckpoint)
    checkpoint_callback = Mock()
    expected_result = Mock()
    initial_states = [Mock(), Mock()]

    with (
        patch("tmd.fe.rbfe.SingleTopology"),
        patch("tmd.fe.rbfe.bisection_lambda_schedule", return_value=np.array([0.0, 1.0])),
        patch("tmd.fe.rbfe.setup_initial_states", return_value=initial_states),
        patch("tmd.fe.rbfe.get_mol_name", return_value="mol"),
        patch("tmd.fe.rbfe.estimate_relative_free_energy_bisection_hrex_impl", return_value=expected_result) as impl,
    ):
        result = estimate_relative_free_energy_bisection_hrex(
            Mock(),
            Mock(),
            np.empty((0, 2), dtype=np.int32),
            Mock(),
            None,
            md_params=md_params,
            n_windows=2,
            resume_state=resume_state,
            checkpoint_interval_frames=2,
            checkpoint_callback=checkpoint_callback,
        )

    assert result is expected_result
    assert impl.call_args.kwargs == {
        "resume_state": resume_state,
        "checkpoint_interval_frames": 2,
        "checkpoint_callback": checkpoint_callback,
    }


@pytest.mark.parametrize("leg_name", ["vacuum", "solvent", "complex"])
def test_rbfe_leg_forwards_checkpoint_arguments(leg_name):
    md_params = make_hrex_md_params()
    resume_state = Mock(spec=HREXCheckpoint)
    checkpoint_callback = Mock()
    forcefield = SimpleNamespace(water_ff=Mock(), protein_ff=Mock())
    mol_a = Mock()
    mol_b = Mock()
    core = np.empty((0, 2), dtype=np.int32)
    expected_result = Mock()
    host_config = Mock()

    with (
        patch("tmd.fe.rbfe.estimate_relative_free_energy_bisection_or_hrex", return_value=expected_result) as estimate,
        patch("tmd.fe.rbfe.builders.build_water_system", return_value=host_config),
        patch("tmd.fe.rbfe.builders.build_protein_system", return_value=host_config),
        patch("tmd.fe.rbfe.setup_optimized_host", return_value=host_config),
    ):
        checkpoint_kwargs = {
            "resume_state": resume_state,
            "checkpoint_interval_frames": 2,
            "checkpoint_callback": checkpoint_callback,
        }
        if leg_name == "vacuum":
            result = run_vacuum(mol_a, mol_b, core, forcefield, None, md_params=md_params, **checkpoint_kwargs)
        elif leg_name == "solvent":
            result, returned_host_config = run_solvent(
                mol_a, mol_b, core, forcefield, None, md_params=md_params, **checkpoint_kwargs
            )
            assert returned_host_config is host_config
        else:
            result, returned_host_config = run_complex(
                mol_a, mol_b, core, forcefield, "protein.pdb", md_params=md_params, **checkpoint_kwargs
            )
            assert returned_host_config is host_config

    assert result is expected_result
    assert estimate.call_args.kwargs["resume_state"] is resume_state
    assert estimate.call_args.kwargs["checkpoint_interval_frames"] == 2
    assert estimate.call_args.kwargs["checkpoint_callback"] is checkpoint_callback


def test_hrex_implementation_invokes_checkpoint_callback_synchronously_and_resumes():
    md_params = make_hrex_md_params()
    resume_state = Mock(spec=HREXCheckpoint)
    resume_state.initial_states_hrex = None
    resume_state.completed_frames = None
    checkpoints = [make_production_checkpoint(completed_frames) for completed_frames in (1, 2)]
    callback_calls = []
    bisection_output = make_stub_bisection_output()
    pair_bar_result = Mock()
    production_trajectories = [Mock(), Mock()]
    hrex_diagnostics = Mock(transition_matrix=Mock(), cumulative_replica_state_counts=Mock())
    final_result = (pair_bar_result, production_trajectories, hrex_diagnostics, None)

    def run_checkpointing_hrex(initial_states_hrex, *args, **kwargs):
        assert kwargs["resume_state"] is resume_state
        assert kwargs["checkpoint_interval_frames"] == 1
        # The implementation attaches the schedule and bisection report to each yielded checkpoint.
        expected = [
            replace(checkpoint, initial_states_hrex=initial_states_hrex, bisection_results=bisection_output[0])
            for checkpoint in checkpoints
        ]
        yield checkpoints[0]
        # callback_calls[0] is the pre-production checkpoint carrying the freshly computed schedule.
        assert callback_calls[0].completed_frames is None
        assert callback_calls[0].initial_states_hrex is initial_states_hrex
        assert callback_calls[0].bisection_results is bisection_output[0]
        assert callback_calls[1:] == expected[:1]
        yield checkpoints[1]
        assert callback_calls[1:] == expected
        return final_result

    with (
        patch("tmd.fe.rbfe.run_sims_bisection", return_value=bisection_output),
        patch("tmd.fe.rbfe.run_sims_hrex_iter", side_effect=run_checkpointing_hrex),
        patch("tmd.fe.rbfe.make_pair_bar_plots", return_value=Mock()),
        patch("tmd.fe.rbfe.plot_as_png_fxn", return_value=b"plot"),
    ):
        result = estimate_relative_free_energy_bisection_hrex_impl(
            temperature=300.0,
            lambda_min=0.0,
            lambda_max=1.0,
            md_params=md_params,
            n_windows=2,
            make_initial_state_fn=Mock(),
            optimize_initial_state_fn=Mock(),
            combined_prefix="checkpoint",
            resume_state=resume_state,
            checkpoint_interval_frames=1,
            checkpoint_callback=callback_calls.append,
        )

    assert len(callback_calls) == 3
    assert result.final_result is pair_bar_result
    assert result.trajectories is production_trajectories
    assert result.hrex_diagnostics is hrex_diagnostics


def test_hrex_implementation_fires_checkpoint_callback_once_after_bisection_on_fresh_run():
    md_params = make_hrex_md_params()
    checkpoints = [make_production_checkpoint(completed_frames) for completed_frames in (1, 2)]
    callback_calls = []
    bisection_output = make_stub_bisection_output()
    pair_bar_result = Mock()
    production_trajectories = [Mock(), Mock()]
    hrex_diagnostics = Mock(transition_matrix=Mock(), cumulative_replica_state_counts=Mock())
    final_result = (pair_bar_result, production_trajectories, hrex_diagnostics, None)

    def run_checkpointing_hrex(initial_states_hrex, *args, **kwargs):
        yield checkpoints[0]
        yield checkpoints[1]
        return final_result

    with (
        patch("tmd.fe.rbfe.run_sims_bisection", return_value=bisection_output),
        patch("tmd.fe.rbfe.run_sims_hrex_iter", side_effect=run_checkpointing_hrex),
        patch("tmd.fe.rbfe.make_pair_bar_plots", return_value=Mock()),
        patch("tmd.fe.rbfe.plot_as_png_fxn", return_value=b"plot"),
    ):
        estimate_relative_free_energy_bisection_hrex_impl(
            temperature=300.0,
            lambda_min=0.0,
            lambda_max=1.0,
            md_params=md_params,
            n_windows=2,
            make_initial_state_fn=Mock(),
            optimize_initial_state_fn=Mock(),
            combined_prefix="checkpoint",
            resume_state=None,
            checkpoint_interval_frames=1,
            checkpoint_callback=callback_calls.append,
        )

    assert len(callback_calls) == 3
    first_call = callback_calls[0]
    assert first_call.completed_frames is None
    assert first_call.hrex is None
    assert first_call.iterated_u_kln is None
    assert first_call.replica_idx_by_state_by_iter == []
    assert first_call.fraction_accepted_by_pair_by_iter == []
    assert first_call.water_sampler_proposals_by_state_by_iter == []
    assert [s.lamb for s in first_call.initial_states_hrex] == [0.0, 1.0]
    assert first_call.bisection_results is bisection_output[0]
    # The remaining calls are the per-interval production checkpoints yielded by run_sims_hrex_iter, each
    # with completed_frames set — confirming the schedule-only checkpoint fired strictly before production.
    assert [call.completed_frames for call in callback_calls[1:]] == [1, 2]


def test_hrex_implementation_propagates_stop_iteration_from_pre_production_checkpoint_callback():
    md_params = make_hrex_md_params()
    bisection_output = make_stub_bisection_output()
    checkpoint_callback = Mock(side_effect=StopIteration("pre-production callback stopped"))

    with (
        patch("tmd.fe.rbfe.run_sims_bisection", return_value=bisection_output),
        patch("tmd.fe.rbfe.run_sims_hrex_iter") as run_sims_hrex_iter,
        pytest.raises(StopIteration, match="pre-production callback stopped"),
    ):
        estimate_relative_free_energy_bisection_hrex_impl(
            temperature=300.0,
            lambda_min=0.0,
            lambda_max=1.0,
            md_params=md_params,
            n_windows=2,
            make_initial_state_fn=Mock(),
            optimize_initial_state_fn=Mock(),
            combined_prefix="checkpoint",
            resume_state=None,
            checkpoint_interval_frames=1,
            checkpoint_callback=checkpoint_callback,
        )

    checkpoint_callback.assert_called_once()
    run_sims_hrex_iter.assert_not_called()


def test_hrex_implementation_propagates_stop_iteration_from_checkpoint_callback():
    md_params = make_hrex_md_params()
    # A resume state with a locked schedule skips bisection, so the only checkpoint callback call is the
    # production one made from the generator below.
    bisection_output = make_stub_bisection_output()
    resume_state = make_production_checkpoint(
        1, initial_states_hrex=[StubInitialState(0.0)], bisection_results=bisection_output[0]
    )
    checkpoint = replace(resume_state, completed_frames=2)
    checkpoint_callback = Mock(side_effect=StopIteration("callback stopped"))

    def run_checkpointing_hrex(*args, **kwargs):
        yield checkpoint
        raise AssertionError("checkpoint callback did not stop HREX result consumption")

    with (
        patch("tmd.fe.rbfe.run_sims_bisection", return_value=bisection_output),
        patch("tmd.fe.rbfe.run_sims_hrex_iter", side_effect=run_checkpointing_hrex),
        patch("tmd.fe.rbfe.pickle.dump"),
        patch("builtins.open", mock_open()),
        pytest.raises(StopIteration, match="callback stopped"),
    ):
        estimate_relative_free_energy_bisection_hrex_impl(
            temperature=300.0,
            lambda_min=0.0,
            lambda_max=1.0,
            md_params=md_params,
            n_windows=2,
            make_initial_state_fn=Mock(),
            optimize_initial_state_fn=Mock(),
            combined_prefix="checkpoint",
            resume_state=resume_state,
            checkpoint_interval_frames=1,
            checkpoint_callback=checkpoint_callback,
        )

    checkpoint_callback.assert_called_once_with(checkpoint)


def test_hrex_implementation_skips_bisection_and_reuses_saved_schedule_on_resume():
    md_params = make_hrex_md_params()
    resume_state = make_production_checkpoint(
        2, initial_states_hrex=[StubInitialState(0.3)], bisection_results=["sentinel-report"]
    )
    callback_calls = []
    pair_bar_result = Mock()
    production_trajectories = [Mock(), Mock()]
    hrex_diagnostics = Mock(transition_matrix=Mock(), cumulative_replica_state_counts=Mock())
    final_result = (pair_bar_result, production_trajectories, hrex_diagnostics, None)

    def run_checkpointing_hrex(initial_states_hrex, *args, **kwargs):
        # The implementation must pass the resumed checkpoint's own saved schedule through unchanged, not a
        # freshly computed one.
        assert initial_states_hrex is resume_state.initial_states_hrex
        yield make_production_checkpoint(3)
        return final_result

    with (
        patch("tmd.fe.rbfe.run_sims_bisection") as run_sims_bisection,
        patch("tmd.fe.rbfe.run_sims_hrex_iter", side_effect=run_checkpointing_hrex),
        patch("tmd.fe.rbfe.make_pair_bar_plots", return_value=Mock()),
        patch("tmd.fe.rbfe.plot_as_png_fxn", return_value=b"plot"),
    ):
        result = estimate_relative_free_energy_bisection_hrex_impl(
            temperature=300.0,
            lambda_min=0.0,
            lambda_max=1.0,
            md_params=md_params,
            n_windows=2,
            make_initial_state_fn=Mock(),
            optimize_initial_state_fn=Mock(),
            combined_prefix="checkpoint",
            resume_state=resume_state,
            checkpoint_interval_frames=1,
            checkpoint_callback=callback_calls.append,
        )

    run_sims_bisection.assert_not_called()
    assert result.intermediate_results is resume_state.bisection_results
    assert len(callback_calls) == 1
    assert not any(call.completed_frames is None for call in callback_calls)


def test_hrex_implementation_rejects_production_state_without_a_saved_schedule():
    md_params = make_hrex_md_params()
    resume_state = make_production_checkpoint(2)

    with pytest.raises(ValueError, match="no saved lambda schedule"):
        estimate_relative_free_energy_bisection_hrex_impl(
            temperature=300.0,
            lambda_min=0.0,
            lambda_max=1.0,
            md_params=md_params,
            n_windows=2,
            make_initial_state_fn=Mock(),
            optimize_initial_state_fn=Mock(),
            combined_prefix="checkpoint",
            resume_state=resume_state,
            checkpoint_interval_frames=1,
            checkpoint_callback=Mock(),
        )
