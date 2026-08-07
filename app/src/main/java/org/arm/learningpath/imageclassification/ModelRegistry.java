package org.arm.learningpath.imageclassification;

import java.util.List;
import java.util.Locale;

final class ModelRegistry {
    private static final List<ModelDescriptor> MODELS = List.of(
            new ModelDescriptor(
                    "deit-tiny-litert",
                    "DEiT Tiny",
                    ModelTask.FIXED_LABEL_CLASSIFICATION,
                    ModelDescriptor.ModelRuntime.LITERT,
                    "deit-tiny-int8-litert.tflite",
                    LiteRtClassifier.PreprocessingProfile.IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "google-vit-litert",
                    "Google ViT",
                    ModelTask.FIXED_LABEL_CLASSIFICATION,
                    ModelDescriptor.ModelRuntime.LITERT,
                    "vit-base-int8-litert.tflite",
                    LiteRtClassifier.PreprocessingProfile.SYMMETRIC_DIRECT_RESIZE
            ),
            new ModelDescriptor(
                    "mobilenet-v3-small-litert",
                    "MobileNetV3 Small",
                    ModelTask.FIXED_LABEL_CLASSIFICATION,
                    ModelDescriptor.ModelRuntime.LITERT,
                    "mobilenet-v3-small-int8-litert.tflite",
                    LiteRtClassifier.PreprocessingProfile.IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "swin-tiny-litert",
                    "Swin Tiny",
                    ModelTask.FIXED_LABEL_CLASSIFICATION,
                    ModelDescriptor.ModelRuntime.LITERT,
                    "swin-tiny-int8-litert.tflite",
                    LiteRtClassifier.PreprocessingProfile.IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "timm-vit-litert",
                    "timm ViT",
                    ModelTask.FIXED_LABEL_CLASSIFICATION,
                    ModelDescriptor.ModelRuntime.LITERT,
                    "vit-base-timm-int8-litert.tflite",
                    LiteRtClassifier.PreprocessingProfile.SYMMETRIC_CROP_232
            ),
            new ModelDescriptor(
                    "clip-vit-b32-executorch",
                    "CLIP ViT-B/32 INT8",
                    ModelTask.DESCRIPTION_MATCHING,
                    ModelDescriptor.ModelRuntime.EXECUTORCH,
                    "clip-vit-base-patch32-int8-executorch.pte",
                    null
            )
    );

    private ModelRegistry() {
    }

    static ModelDescriptor forFileName(String fileName) {
        for (ModelDescriptor descriptor : MODELS) {
            if (descriptor.fileName().equalsIgnoreCase(fileName)) {
                return descriptor;
            }
        }
        return null;
    }

    static ModelDescriptor forId(String id) {
        for (ModelDescriptor descriptor : MODELS) {
            if (descriptor.id().equals(id)) {
                return descriptor;
            }
        }
        return null;
    }

    static ModelDescriptor forStoredName(String storedName) {
        String normalized = storedName.toLowerCase(Locale.ROOT);
        for (ModelDescriptor descriptor : MODELS) {
            String fileStem = descriptor.fileName().replaceFirst("\\.[^.]+$", "")
                    .toLowerCase(Locale.ROOT);
            if (normalized.contains(fileStem)
                    || normalized.contains(descriptor.displayName().toLowerCase(Locale.ROOT))) {
                return descriptor;
            }
        }
        return null;
    }

    static String supportedFiles() {
        StringBuilder text = new StringBuilder();
        for (ModelDescriptor descriptor : MODELS) {
            if (text.length() > 0) {
                text.append(", ");
            }
            text.append(descriptor.fileName());
        }
        return text.toString();
    }
}
