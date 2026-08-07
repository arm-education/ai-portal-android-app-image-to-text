# Android App - Image Classification
## This is an example learning app that uses image classification models from the Arm AI Portal.

This application is provided in relation to the following learning path: https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/ai-portal-mobile-image-classification. It is intended for learning purposes only, to understand how models can be executed on devices, not as a reference production-quality app. It is provided under the [Arm Education End User License Agreement](LICENSE.md).

This Android application runs Arm-optimized image models locally on an Arm64 phone or emulator. It provides two launch workflows:

- **Quick Identify** runs a fixed-label ImageNet classifier with LiteRT and XNNPACK. Several different models can be used for this mode.
- **Custom Match** runs a CLIP Vision Encoder model with ExecuTorch and XNNPACK and compares a photo with descriptions entered by the user.

The application imports model binaries at run time, so the model files are not stored in the Android application package (APK).

## Requirements

- Android Studio with Android SDK 35
- Java 17, supplied by Android Studio
- An Arm64 Android device running Android 9, API 28, or later
- One supported model file downloaded from the Arm AI Portal

The Gradle build includes only the `arm64-v8a` application binary interface (ABI).

## Supported launch models

Keep each downloaded filename unchanged. The launch registry uses the filename to select the task, runtime, and preprocessing profile, then validates the model contract before activation.

| Model | Runtime | Import this file |
| --- | --- | --- |
| [DEiT Tiny LiteRT](https://huggingface.co/Arm/deit-tiny-int8-litert) | LiteRT | `facebook__deit-tiny-patch16-224_litert_optimized.tflite` |
| [Google ViT LiteRT](https://huggingface.co/Arm/vit-base-int8-litert) | LiteRT | `google__vit-base-patch16-224_android_litert_optimized.tflite` |
| [MobileNetV3 Small LiteRT](https://huggingface.co/Arm/mobilenet-v3-small-int8-litert) | LiteRT | `mobilenet_v3_small_android_litert_optimized.tflite` |
| [Swin Tiny LiteRT](https://huggingface.co/Arm/swin-tiny-int8-litert) | LiteRT | `microsoft__swin-tiny-patch4-window7-224_android_litert_optimized.tflite` |
| [timm ViT LiteRT](https://huggingface.co/Arm/vit-base-timm-int8-litert) | LiteRT | `timm__vit_base_patch16_224.augreg_in21k_ft_in1k_android_litert_optimized.tflite` |
| [CLIP ViT-B/32 ExecuTorch](https://huggingface.co/Arm/clip-vit-base-patch32-int8-xnnpack-executorch) | ExecuTorch | `optimized.pte` |

## Open and run the application

1. Clone or download this repository.
2. Open the repository root in Android Studio.
3. Wait for Gradle sync to finish.
4. Connect an Arm64 Android phone or start an Arm64 emulator.
5. Select the `app` configuration and run it.
6. Select **Add classifier model** or **Add CLIP model** and choose the matching optimized file.
7. Choose a photo and run **Quick Identify** or **Custom Match**.

The application stores the selected model in its private files directory. Clearing application data or uninstalling the application removes imported models.

## Application architecture

`ModelRegistry.java` contains the launch model descriptors. Each descriptor records:

- The expected filename and display name.
- The user-facing task.
- The LiteRT or ExecuTorch runtime.
- The LiteRT preprocessing profile, when applicable.

`ModelImporter.java` resolves the descriptor, copies the file into private storage, and asks `ModelRunnerRegistry.java` to validate the registered runtime adapter before activating the model.

`ModelRunnerRegistry.java` provides two launch runner factories:

- `LiteRtClassifier` for fixed-label classification.
- `ClipModel` and `ClipTokenizer` for CLIP description matching.

The user interface works with the shared `ModelRunner` interface and does not construct runtime-specific model loaders directly.

## LiteRT preprocessing profiles

| Profile | Resize and crop | Normalization | Current models |
| --- | --- | --- | --- |
| `IMAGENET_CROP_256` | Resize the shorter edge to 256, then center-crop to the model input size | ImageNet mean and standard deviation | DEiT Tiny, MobileNetV3 Small, Swin Tiny |
| `SYMMETRIC_DIRECT_RESIZE` | Resize directly to the model input size | Mean `0.5` and standard deviation `0.5` for every RGB channel | Google ViT |
| `SYMMETRIC_CROP_232` | Resize the shorter edge to 232, then center-crop to the model input size | Mean `0.5` and standard deviation `0.5` for every RGB channel | timm ViT |

Another classifier can reuse a profile only when its resize, crop, layout, input type, normalization, output, and label order match.

## Arm CPU acceleration

The LiteRT path enables XNNPACK, and the CLIP program uses the ExecuTorch XNNPACK backend. The Arm64 native libraries in LiteRT `2.1.6` and ExecuTorch `1.3.1` contain SME2 code paths. LiteRT `2.1.6` also packages KleidiAI.

On an SME2-capable phone, XNNPACK can select a compatible KleidiAI SME2 kernel automatically for individual operations. Other operations use another compatible XNNPACK path.

## Build from the command line

Set `ANDROID_HOME` to your Android SDK and `JAVA_HOME` to a Java 17 installation, then run:

```bash
./gradlew assembleDebug lint
```

The debug APK is written under `app/build/outputs/apk/debug/`.

## License

This project is provided under the [Arm Education End User License Agreement](LICENSE.md).
