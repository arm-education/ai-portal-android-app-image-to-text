package org.arm.learningpath.imageclassification;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

final class ModelRegistry {
    private static final List<ModelDescriptor> BUILT_IN_MODELS = List.of(
            new ModelDescriptor(
                    "deit-tiny-litert",
                    "DEiT Tiny",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "deit-tiny-int8-litert.tflite"
            ),
            new ModelDescriptor(
                    "google-vit-litert",
                    "Google ViT",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "vit-base-int8-litert.tflite"
            ),
            new ModelDescriptor(
                    "mobilenet-v3-small-litert",
                    "MobileNetV3 Small",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "mobilenet-v3-small-int8-litert.tflite"
            ),
            new ModelDescriptor(
                    "swin-tiny-litert",
                    "Swin Tiny",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "swin-tiny-int8-litert.tflite"
            ),
            new ModelDescriptor(
                    "timm-vit-litert",
                    "timm ViT",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "vit-base-timm-int8-litert.tflite"
            ),
            new ModelDescriptor(
                    "clip-vit-b32-executorch",
                    "CLIP ViT-B/32 INT8",
                    ExecuTorchClipAdapter.ID,
                    "ExecuTorch",
                    "clip-vit-base-patch32-int8-executorch.pte"
            )
    );
    private static final List<ModelDescriptor> MODELS = createModels();

    private ModelRegistry() {
    }

    private static List<ModelDescriptor> createModels() {
        List<ModelDescriptor> models = new ArrayList<>(BUILT_IN_MODELS);
        models.addAll(GeneratedAdapterRegistry.models());
        return List.copyOf(models);
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

    static List<ModelDescriptor> forAdapter(String adapterId) {
        List<ModelDescriptor> matching = new ArrayList<>();
        for (ModelDescriptor descriptor : MODELS) {
            if (descriptor.adapterId().equals(adapterId)) {
                matching.add(descriptor);
            }
        }
        return List.copyOf(matching);
    }

    static String supportedFiles() {
        StringBuilder files = new StringBuilder();
        for (ModelDescriptor descriptor : MODELS) {
            if (files.length() > 0) {
                files.append(", ");
            }
            files.append(descriptor.fileName());
        }
        return files.toString();
    }
}
