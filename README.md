# Photo Insight Android application

This example application accompanies the [Arm Learning Path for running image classification models from the Arm AI Portal](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/ai-portal-mobile-image-classification). It is intended for learning how models run on devices and is not a reference production application. It is provided under the [Arm Education End User License Agreement](LICENSE.md).

This Android application runs Arm-optimized image models locally on an Arm64 phone or emulator. It provides two workflows:

- **Quick Identify** runs a fixed-label ImageNet classifier with LiteRT and XNNPACK. Several different models can be used for this mode.
- **Custom Match** runs a CLIP model with ExecuTorch and XNNPACK and compares a photo with descriptions entered by the user.

The application imports model binaries at run time, so the model files are not stored in the Android application package (APK).

## Requirements

- Android Studio with Android SDK 35
- Java 17, supplied by Android Studio
- An Arm64 Android device running Android 9, API 28, or later
- One supported model file downloaded from the Arm AI Portal

## Supported launch models

Keep each downloaded filename unchanged. The launch registry uses the filename to select the task, runtime, and preprocessing profile, then validates the model contract before activation.

| Model | Runtime | Import this file |
| --- | --- | --- |
| [DEiT Tiny LiteRT](https://huggingface.co/Arm/deit-tiny-int8-litert) | LiteRT | `deit-tiny-int8-litert.tflite` |
| [Google ViT LiteRT](https://huggingface.co/Arm/vit-base-int8-litert) | LiteRT | `vit-base-int8-litert.tflite` |
| [MobileNetV3 Small LiteRT](https://huggingface.co/Arm/mobilenet-v3-small-int8-litert) | LiteRT | `mobilenet-v3-small-int8-litert.tflite` |
| [Swin Tiny LiteRT](https://huggingface.co/Arm/swin-tiny-int8-litert) | LiteRT | `swin-tiny-int8-litert.tflite` |
| [timm ViT LiteRT](https://huggingface.co/Arm/vit-base-timm-int8-litert) | LiteRT | `vit-base-timm-int8-litert.tflite` |
| [CLIP ViT-B/32 ExecuTorch](https://huggingface.co/Arm/clip-vit-base-patch32-int8-xnnpack-executorch) | ExecuTorch | `clip-vit-base-patch32-int8-executorch.pte` |

## Download a model

Create a Python virtual environment and install the Hugging Face Hub package:

On macOS or Linux:

```bash
python3 -m venv .hf-venv
source .hf-venv/bin/activate
python -m pip install --upgrade huggingface_hub
```

On Windows PowerShell:

```powershell
py -m venv .hf-venv
.\.hf-venv\Scripts\Activate.ps1
python -m pip install --upgrade huggingface_hub
```

Set the repository ID for one of the supported models, then run the included download script. The `--print-path` option returns the downloaded file path for later commands:

On macOS or Linux:

```bash
MODEL_ID="Arm/mobilenet-v3-small-int8-litert"
MODEL_FILE="$(python download_model.py \
  --repo-id "$MODEL_ID" \
  --print-path)"

printf 'Model file: %s\n' "$MODEL_FILE"
```

On Windows PowerShell:

```powershell
$MODEL_ID = "Arm/mobilenet-v3-small-int8-litert"
$MODEL_FILE = python download_model.py `
  --repo-id "$MODEL_ID" `
  --print-path

Write-Output "Model file: $MODEL_FILE"
```

The script downloads only the optimized `.tflite` or `.pte` file accepted by the application. It stores the file under `models/` and preserves the filename used by `ModelRegistry.java`.

Copy the downloaded model to the Android **Downloads** directory through ADB:

```console
adb push "$MODEL_FILE" /sdcard/Download/
```

## Open and run the application

1. Clone or download this repository.
2. Open the repository root in Android Studio.
3. Wait for Gradle sync to finish.
4. Connect an Arm64 Android phone or start an Arm64 emulator.
5. Select the `app` configuration and run it.
6. Select **Add classifier model** or **Add CLIP model** and choose the matching optimized file.
7. Select **Choose a photo**, then run **Quick Identify** or **Custom Match**.

The application does not include a sample photo. Each user selects the image they want to analyze from the Android document picker.

The application stores the selected model in its private files directory. Clearing application data or uninstalling the application removes imported models.

## License

This project is provided under the [Arm Education End User License Agreement](LICENSE.md).
