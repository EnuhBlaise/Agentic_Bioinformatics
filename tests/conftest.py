import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioagent.config import load_settings  # noqa: E402
from bioagent.demo import make_dataset  # noqa: E402


@pytest.fixture
def settings_factory(tmp_path):
    def make(**over):
        base = {"workspace": str(tmp_path / "runs"), "state_dir": str(tmp_path / "state"),
                "memory": {"path": str(tmp_path / "state" / "memory")},
                "tools_dir": str(ROOT / "tools"), "workflows_dir": str(ROOT / "workflows")}
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return load_settings(ROOT / "configs" / "mock.yaml", overrides=base)
    return make


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "data", n_genes=1500)


@pytest.fixture
def confounded_dataset(tmp_path):
    return make_dataset(tmp_path / "data_fc", n_genes=1500, fully_confounded=True)
