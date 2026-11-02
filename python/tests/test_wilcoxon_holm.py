import numpy as np
import pytest

from experiments.rl_wilcoxon_pairwise import holm


def test_holm_step_down_and_nan():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(holm([0.01, float("nan")])[1])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
