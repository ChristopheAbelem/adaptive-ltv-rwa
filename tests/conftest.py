import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_repo_root():
    """Data paths are relative to the repo root."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(cwd)
