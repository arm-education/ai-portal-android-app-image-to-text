#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


SUPPORTED_MODELS = {
    "Arm/deit-tiny-int8-litert": "deit-tiny-int8-litert.tflite",
    "Arm/vit-base-int8-litert": "vit-base-int8-litert.tflite",
    "Arm/mobilenet-v3-small-int8-litert": "mobilenet-v3-small-int8-litert.tflite",
    "Arm/swin-tiny-int8-litert": "swin-tiny-int8-litert.tflite",
    "Arm/vit-base-timm-int8-litert": "vit-base-timm-int8-litert.tflite",
    "Arm/clip-vit-base-patch32-int8-xnnpack-executorch": (
        "clip-vit-base-patch32-int8-executorch.pte"
    ),
}


def model_directory(output_directory: Path, model_id: str) -> Path:
    return output_directory / model_id.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a model file supported by the Photo Insight Android app."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Parent directory for downloaded models (default: models)",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print only the downloaded model path to standard output",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    filename = SUPPORTED_MODELS[args.repo_id]
    destination = model_directory(args.output_dir, args.repo_id)

    output_stream = sys.stderr if args.print_path else sys.stdout
    print(
        f"Downloading {args.repo_id}/{filename} to {destination} ...",
        file=output_stream,
    )
    downloaded_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=destination,
        )
    )

    if not downloaded_path.is_file():
        raise SystemExit(f"The downloaded model file was not found: {downloaded_path}")

    resolved_path = downloaded_path.resolve()
    if args.print_path:
        print(resolved_path)
    else:
        print(f"Model file: {resolved_path}")


if __name__ == "__main__":
    main()
