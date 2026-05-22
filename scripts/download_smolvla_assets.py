"""Download/stage the SmolVLA checkpoint from Hugging Face.

This stages the proposal-compatible lightweight VLA asset without requiring the full LeRobot runtime to be installed.
The runtime can be installed later with `pip install "lerobot[smolvla]"`; if that install is unavailable, this script
still records a manifest showing the checkpoint files are local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.vla import DEFAULT_SMOLVLA_REPO_ID, check_smolvla_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Download SmolVLA assets for the pre-grasp refusal project.")
    parser.add_argument("--repo-id", default=DEFAULT_SMOLVLA_REPO_ID)
    parser.add_argument("--output-dir", default="logs/hsi_pregrasp/vla/smolvla_base")
    parser.add_argument("--metadata-only", action="store_true", help="Skip model.safetensors.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(args.repo_id, files_metadata=True)
    files = {sibling.rfilename: sibling.size for sibling in info.siblings}
    allow_patterns = [
        "README.md",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    if not args.metadata_only:
        allow_patterns.append("model.safetensors")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )

    status = check_smolvla_assets(snapshot_path, repo_id=args.repo_id)
    manifest = {
        "repo_id": args.repo_id,
        "snapshot_path": str(snapshot_path),
        "metadata_only": args.metadata_only,
        "files": files,
        "status": status.as_dict(),
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
