import pytest


@pytest.fixture
def fast():
    """Settings every quick run shares: Reference platform, no warm-up."""
    return dict(platform="Reference", minimize=False, equilibration_ns=0.0,
                exchange_interval_steps=50, friction_per_ps=5.0,
                random_seed=7, save_selection="all")
