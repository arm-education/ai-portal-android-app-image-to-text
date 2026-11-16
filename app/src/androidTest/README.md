# Model device tests

`ModelSupportTest` checks unique registry IDs/filenames, imports through the real
`ModelImporter` using a fixture content provider, and runs two different photos
through each staged model. It checks for Samoyed and cat labels in the top five,
then closes the runner and imports the next model. CLIP regression checks rank
dog and cat descriptions on their corresponding images.

Build the test APK with `./gradlew :app:assembleDebug :app:assembleDebugAndroidTest`.
Install both APKs with `adb install -r`, then stage fixtures in the target app's
`files/model-test-inputs` directory. Use a separate directory from installed models;
the tests use the app cache for imports and leave installed models and state alone.

Download all eight vivo models with `download_model.py`, keeping its output names.
For each downloaded model, copy it through a shell-readable temporary location:

```console
adb push "$MODEL_FILE" /data/local/tmp/
adb shell run-as org.arm.learningpath.imageclassification mkdir -p files/model-test-inputs
adb shell run-as org.arm.learningpath.imageclassification cp "/data/local/tmp/$(basename "$MODEL_FILE")" files/model-test-inputs/
```

Stage `dog.jpg` from [PyTorch Hub](https://github.com/pytorch/hub/blob/master/images/dog.jpg)
and a Persian cat photo as `cat.jpg` the same way. The validation run used the
[Learning Paths cat fixture](https://github.com/ArmDeveloperEcosystem/arm-learning-paths/blob/main/content/learning-paths/embedded-and-microcontrollers/alif-image-classification/cat.jpg).
Additional registered models can be copied into the fixture directory to include
them in regression checks, using the unchanged registered filenames.

Run the tests:

```console
adb shell am instrument -w org.arm.learningpath.imageclassification.test/android.test.InstrumentationTestRunner
adb logcat -d -s ModelSupportTest:I
```

The suite requires all eight vivo models. It logs each model's two results and
reports a pass count. Two-photo checks establish integration and plausible task
output, not ImageNet accuracy or benchmark parity. The test APK uses Android's
framework test libraries; they are not added to the production APK.

## Preprocessing sources

The eight Arm Hugging Face repositories linked from the README provide the
`.pte` binaries, model cards, and benchmark input sizes. Wide ResNet-50,
ViT-B/16, and GoogLeNet additionally publish `config.yaml` and
`imagenet_classes.json`; their label files match the bundled labels.

For the other exports, preprocessing follows the linked original model:

- [Inception V3 config](https://huggingface.co/timm/inception_v3.tv_in1k/blob/main/config.json): bicubic, crop fraction 0.875, mean/std 0.5; 299-pixel exported input gives a shorter edge of 341.
- [EfficientNet-B4 config](https://huggingface.co/timm/efficientnet_b4.ra2_in1k/blob/main/config.json): bicubic, crop fraction 1.0, ImageNet normalization; use the export's 384-pixel test size.
- [DeiT-Tiny processor](https://huggingface.co/facebook/deit-tiny-patch16-224/blob/main/preprocessor_config.json): direct 224-pixel resize, mean/std 0.5, [default bilinear interpolation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/vit/image_processing_vit.py). The repository declares `model_type: vit`, so the automatic processor selects ViT preprocessing, despite the DeiT model name.
- [Swin-T processor](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/blob/main/preprocessor_config.json): direct 224-pixel bicubic resize, ImageNet normalization.
- [SqueezeNet 1.0 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_0.html): bilinear shorter-edge 256 resize, center crop 224, ImageNet normalization.

The optimized repositories do not all document the complete export-time transform.
These upstream settings and device checks do not prove reproduction of the Portal
accuracy measurements. Resize implementations can differ in rounding and antialiasing.
