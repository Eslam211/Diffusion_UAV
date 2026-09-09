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
        "cql",
        build_agent,
        REPO_ROOT,
        {
            "cql_alpha": 1.0,
            "entropy_alpha": 0.2,
            "cql_num_random": 10,
        },
    )
