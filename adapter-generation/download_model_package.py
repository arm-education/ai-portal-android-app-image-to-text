#!/usr/bin/env python3

import argparse
from pathlib import Path


def model_directory(output_directory: Path, model_id: str) -> Path:
    return output_directory / model_id.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a complete model package for adapter generation."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    destination = model_directory(args.output_dir, args.repo_id)
    downloaded = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=destination,
    )
    print(Path(downloaded).resolve())


if __name__ == "__main__":
    main()
