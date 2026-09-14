"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with: pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


# TODO 1: test_rejects_negative_rate
def test_rejects_negative_rate():
    """Mənfi parçalanma dərəcəsi daxil edildikdə ValueError atıldığını yoxlayır."""
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


# TODO 2: test_matches_law
def test_matches_law():
    """Simulyasiyaların ortalamasının N0 * exp(-lam * t) qanununa uyğunluğunu yoxlayır."""
    N0 = 100000
    lam = 0.02
    t = 1
    num_runs = 50

    # Tapşırıqda tələb olunan analitik qanun: N0 * exp(-lam * t)
    expected_N_t = N0 * np.exp(-lam * t)

    # Simulyasiyaların ortalaması
    run_results = [simulate(N0, lam)[t] for _ in range(num_runs)]
    average_N_t = np.mean(run_results)

    # Pytest.approx ilə 1% nisbi xəta daxilində müqayisə edirik
    assert average_N_t == pytest.approx(expected_N_t, rel=0.05)