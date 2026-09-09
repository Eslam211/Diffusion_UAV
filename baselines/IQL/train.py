from pathlib import Path
import sys

METHOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, METHOD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent import build_agent
from common.cli import train_entrypoint


if __name__ == "__main__":
    train_entrypoint(
        "iql",
        build_agent,
        REPO_ROOT,
        {
            "expectile": 0.7,
            "advantage_temperature": 3.0,
            "max_advantage_weight": 100.0,
            "lr": 3e-4,
        },
    )
