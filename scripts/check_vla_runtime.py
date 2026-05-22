"""Check whether staged SmolVLA assets can be used by the local runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.vla import DEFAULT_SMOLVLA_REPO_ID, check_smolvla_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Check SmolVLA asset/runtime status.")
    parser.add_argument("--local-dir", default="logs/hsi_pregrasp/vla/smolvla_base")
    parser.add_argument("--repo-id", default=DEFAULT_SMOLVLA_REPO_ID)
    args = parser.parse_args()

    status = check_smolvla_assets(args.local_dir, repo_id=args.repo_id)
    print(json.dumps(status.as_dict(), indent=2))
    if not status.ready_for_runtime:
        print(
            "\nSmolVLA checkpoint is staged, but runtime is incomplete. "
            "Install LeRobot with `pip install \"lerobot[smolvla]\"` when dependency resolution is available."
        )


if __name__ == "__main__":
    main()
