import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-level",
        action="store",
        default="core_model",
        choices=["core_model", "advanced_model"],
        help="Test level to run: L2, L3",
    )


@pytest.fixture(scope="session")
def run_level(request) -> str:
    """CLI test level for the session (``--run-level``). See CI level docs."""
    return request.config.getoption("--run-level")
