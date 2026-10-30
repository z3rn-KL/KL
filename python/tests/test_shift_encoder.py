import numpy as np
import pytest

from rl.shift.encoder import FEATURE_DIMENSION, FEATURE_NAMES, ShiftEncoder
from tests.test_shift_environment import random_environment

COLUMN = {name: i for i, name in enumerate(FEATURE_NAMES)}


def test_one_row_per_unserved_order_plus_stop():
    env = random_environment(1, n=6)
    env.reset()
    env.step(2)

    actions, rows = ShiftEncoder().encode(env)

    assert rows.shape == (6, FEATURE_DIMENSION)
    assert list(actions) == [0, 1, 3, 4, 5, env.stop_action]
    assert rows[:, COLUMN["is_stop"]].tolist() == [0, 0, 0, 0, 0, 1]
    assert np.isfinite(rows).all()


def test_immediate_gain_is_step_reward_over_deferring():
    env = random_environment(2, n=5)
    env.reset()

    actions, rows = ShiftEncoder().encode(env)
    defer = env.config.defer_penalty

    for action, row in zip(actions[:-1], rows[:-1]):
        env.reset()
        reward = env.step(int(action)).reward

        # Only valid while serving does not end the shift (n > 1).
        assert row[COLUMN["immediate_gain"]] == pytest.approx(
            (reward + defer) / defer, abs=1e-4
        )


def test_stop_row_gain_is_the_terminal_reward():
    env = random_environment(3, n=5)
    env.reset()
    env.step(0)

    actions, rows = ShiftEncoder().encode(env)
    expected = env.step(env.stop_action).reward / env.config.defer_penalty

    assert rows[-1, COLUMN["immediate_gain"]] == pytest.approx(
        expected, abs=1e-4
    )


def test_last_remaining_order_has_no_neighbour_features():
    env = random_environment(4, n=3)
    env.reset()
    env.step(0)
    env.step(1)

    _, rows = ShiftEncoder().encode(env)

    assert rows.shape[0] == 2
    assert rows[0, COLUMN["nearest_remaining_ratio"]] == 0.0
    assert rows[0, COLUMN["mean_remaining_ratio"]] == 0.0
    assert rows[0, COLUMN["earlier_deadline_share"]] == 0.0


def test_late_flag_and_earlier_deadline_share():
    env = random_environment(5, n=6, deadline_scale=0.05)
    env.reset()

    actions, rows = ShiftEncoder().encode(env)

    has = rows[:-1, COLUMN["deadline_valid"]] == 1
    assert (rows[:-1, COLUMN["late_flag"]][has] == 1).all()

    # The order with the earliest deadline has nobody ahead of it.
    deadlines = env.deadline_np
    first = int(np.argmin(deadlines))
    assert rows[first, COLUMN["earlier_deadline_share"]] == 0.0


def test_base_is_the_exact_scaled_immediate_reward():
    from rl.shift.encoder import REWARD_SCALE

    env = random_environment(6, n=6)
    env.reset()

    actions, _, base = ShiftEncoder().encode_with_base(env)

    for action, value in zip(actions[:-1], base[:-1]):
        env.reset()
        assert value == pytest.approx(
            env.step(int(action)).reward * REWARD_SCALE, abs=1e-5
        )

    env.reset()
    assert base[-1] == pytest.approx(
        env.step(env.stop_action).reward * REWARD_SCALE, abs=1e-5
    )


def test_pair_times_match_the_travel_matrix_and_zero_the_stop_row():
    from rl.shift.encoder import TIME_REF_MINUTES

    env = random_environment(7, n=5)
    env.reset()
    env.step(1)

    encoder = ShiftEncoder()
    actions, _ = encoder.encode(env)
    pairs = encoder.pair_times(env, actions)

    orders = actions[:-1]

    assert pairs.shape == (len(actions), len(actions))
    assert pairs[0, 1] == pytest.approx(
        env.travel_np[orders[0] + 1, orders[1] + 1] / TIME_REF_MINUTES, abs=1e-6
    )
    assert (pairs[-1] == 0).all() and (pairs[:, -1] == 0).all()
