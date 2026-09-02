#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path


VERIFIED_MODELS = {
    "Arm/deit-tiny-int8-litert": "facebook__deit-tiny-patch16-224_litert_optimized.tflite",
    "Arm/vit-base-int8-litert": "google__vit-base-patch16-224_android_litert_optimized.tflite",
    "Arm/mobilenet-v3-small-int8-litert": "mobilenet_v3_small_android_litert_optimized.tflite",
    "Arm/swin-tiny-int8-litert": "microsoft__swin-tiny-patch4-window7-224_android_litert_optimized.tflite",
    "Arm/vit-base-timm-int8-litert": "timm__vit_base_patch16_224.augreg_in21k_ft_in1k_android_litert_optimized.tflite",
    "Arm/deit-tiny-int8-xnnpack-executorch": "deit_raspberry_executorch_optimized.pte",
    "Arm/googlenet-int8-xnnpack-executorch-raspberrypi5": (
        "googlenet_raspberry_executorch_optimized.pte"
    ),
    "Arm/inception-v3-int8-xnnpack-executorch-raspberrypi5": (
        "inception_v3_raspberry_executorch_optimized.pte"
    ),
    "Arm/mobilenet-v3-small-int8-xnnpack-executorch": "optimized.pte",
    "Arm/resnet-18-int8-xnnpack-executorch": "resnet-18_raspberry_executorch_optimized.pte",
    "Arm/resnet-50-int8-xnnpack-executorch": "resnet-50_raspberry_executorch_optimized.pte",
    "Arm/shufflenet-v2-x1-0-int8-xnnpack-executorch": (
        "shufflenet_v2_x1_0_raspberry_executorch_optimized.pte"
    ),
    "Arm/squeezenet-1-1-int8-xnnpack-executorch": (
        "squeezenet_1_1_raspberry_executorch_optimized.pte"
    ),
    "Arm/swin-tiny-int8-xnnpack-executorch": "swin_tiny_dynamic_raspberry_executorch_optimized.pte",
    "Arm/vit-base-int8-xnnpack-executorch": "google__vit-base-patch16-224_raspberry_executorch_optimized.pte",
    "Arm/clip-vit-base-patch32-int8-xnnpack-executorch": "optimized.pte",
}

IMPORT_FILENAMES = {
    "Arm/mobilenet-v3-small-int8-xnnpack-executorch": (
        "mobilenet-v3-small-int8-executorch.pte"
    ),
    "Arm/clip-vit-base-patch32-int8-xnnpack-executorch": (
        "clip-vit-base-patch32-int8-executorch.pte"
    ),
}


def model_directory(output_directory: Path, model_id: str) -> Path:
    return output_directory / model_id.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a LiteRT or ExecuTorch model file for Photo Insight."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--filename",
        help="Model filename when the repository contains multiple .tflite or .pte files",
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

    from huggingface_hub import hf_hub_download, snapshot_download

    destination = model_directory(args.output_dir, args.repo_id)
    output_stream = sys.stderr if args.print_path else sys.stdout

    filename = args.filename or VERIFIED_MODELS.get(args.repo_id)
    if filename is not None:
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
        if args.filename is None and args.repo_id in IMPORT_FILENAMES:
            import_path = destination / IMPORT_FILENAMES[args.repo_id]
            shutil.copy2(downloaded_path, import_path)
            downloaded_path = import_path
    else:
        print(f"Downloading {args.repo_id} to {destination} ...", file=output_stream)
        snapshot_path = Path(
            snapshot_download(repo_id=args.repo_id, local_dir=destination)
        )
        candidates = sorted(
            path
            for path in snapshot_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pte", ".tflite"}
        )
        if not candidates:
            raise SystemExit(
                "The repository does not contain a .tflite or .pte model file."
            )
        if len(candidates) > 1:
            choices = ", ".join(path.name for path in candidates)
            raise SystemExit(
                "The repository contains multiple model files. "
                f"Run the command again with --filename. Found: {choices}"
            )
        downloaded_path = candidates[0]

    if not downloaded_path.is_file():
        raise SystemExit(f"The downloaded model file was not found: {downloaded_path}")

    resolved_path = downloaded_path.resolve()
    if args.print_path:
        print(resolved_path)
    else:
        print(f"Model file: {resolved_path}")


if __name__ == "__main__":
    main()
