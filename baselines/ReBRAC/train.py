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
        "rebrac",
        build_agent,
        REPO_ROOT,
        {
            "actor_bc_coef": 1.0,
            "critic_bc_coef": 1.0,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
        },
    )
