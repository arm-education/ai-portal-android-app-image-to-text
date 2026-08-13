#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


PLACEHOLDER_PATTERN = re.compile(
    r"<(?:MODEL|RUNTIME|ADAPTER|PLACEHOLDER|TODO)[A-Z0-9_-]*>"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the files produced for a generated Android vision adapter."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--layout", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    registry = project_root / (
        "app/src/main/java/org/arm/learningpath/imageclassification/"
        "GeneratedAdapterRegistry.java"
    )
    dependencies = project_root / "app/generated-runtime-dependencies.gradle.kts"
    adapter_path = args.adapter if args.adapter.is_absolute() else project_root / args.adapter
    paths = [args.adapter, registry, dependencies]
    if args.layout:
        paths.append(args.layout)

    errors = []
    for path in paths:
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file():
            errors.append(f"Missing file: {resolved}")
            if resolved == adapter_path:
                source_directory = registry.parent
                supplied_adapters = {
                    "ExecuTorchClipAdapter.java",
                    "ExecuTorchImageClassificationAdapter.java",
                    "LiteRtImageClassificationAdapter.java",
                }
                candidates = []
                for candidate in sorted(source_directory.glob("*.java")):
                    if candidate.name in supplied_adapters:
                        continue
                    source = candidate.read_text(encoding="utf-8")
                    if "implements VisionAdapter" in source:
                        candidates.append(candidate.relative_to(project_root).as_posix())
                if candidates:
                    errors.append(
                        "Possible generated adapter files: " + ", ".join(candidates)
                    )
            elif args.layout and resolved == (
                args.layout if args.layout.is_absolute() else project_root / args.layout
            ):
                errors.append(
                    "Omit --layout when the generated adapter does not add a layout file"
                )
            continue
        content = resolved.read_text(encoding="utf-8")
        if PLACEHOLDER_PATTERN.search(content):
            errors.append(f"Unresolved placeholder in {resolved}")

    if adapter_path.is_file():
        adapter_source = adapter_path.read_text(encoding="utf-8")
        if "implements VisionAdapter" not in adapter_source:
            errors.append("The generated class must implement VisionAdapter")
        if "descriptor.runtimeDisplayName()" in adapter_source:
            errors.append(
                "Do not validate runtime compatibility using runtimeDisplayName; "
                "it is presentation text"
            )

    if registry.is_file() and "return List.of();" in registry.read_text(encoding="utf-8"):
        errors.append("GeneratedAdapterRegistry does not register an adapter or model")

    if errors:
        print("Generated adapter validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Generated adapter file validation passed")
    print("Run the Gradle build, lint, and on-device model test next")


if __name__ == "__main__":
    main()
