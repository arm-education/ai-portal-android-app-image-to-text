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
                    "deit-tiny-int8-litert.tflite",
                    LiteRtImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "google-vit-litert",
                    "Google ViT",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "vit-base-int8-litert.tflite",
                    LiteRtImageClassificationAdapter.PROFILE_SYMMETRIC_DIRECT_RESIZE
            ),
            new ModelDescriptor(
                    "mobilenet-v3-small-litert",
                    "MobileNetV3 Small",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "mobilenet-v3-small-int8-litert.tflite",
                    LiteRtImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "swin-tiny-litert",
                    "Swin Tiny",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "swin-tiny-int8-litert.tflite",
                    LiteRtImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "timm-vit-litert",
                    "timm ViT",
                    LiteRtImageClassificationAdapter.ID,
                    "LiteRT",
                    "vit-base-timm-int8-litert.tflite",
                    LiteRtImageClassificationAdapter.PROFILE_SYMMETRIC_CROP_232
            ),
            new ModelDescriptor(
                    "deit-tiny-executorch",
                    "DEiT Tiny",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "deit-tiny-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "googlenet-executorch",
                    "GoogLeNet",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "googlenet-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "inception-v3-executorch",
                    "Inception V3",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "inception-v3-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_342
            ),
            new ModelDescriptor(
                    "mobilenet-v3-small-executorch",
                    "MobileNetV3 Small",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "mobilenet-v3-small-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "resnet-18-executorch",
                    "ResNet-18",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "resnet-18-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "resnet-50-executorch",
                    "ResNet-50",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "resnet-50-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "shufflenet-v2-executorch",
                    "ShuffleNet V2 x1.0",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "shufflenet-v2-x1-0-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "squeezenet-1-1-executorch",
                    "SqueezeNet 1.1",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "squeezenet-1-1-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_256
            ),
            new ModelDescriptor(
                    "swin-tiny-executorch",
                    "Swin Tiny",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "swin-tiny-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_IMAGENET_CROP_232
            ),
            new ModelDescriptor(
                    "vit-base-executorch",
                    "ViT Base",
                    ExecuTorchImageClassificationAdapter.ID,
                    "ExecuTorch",
                    "vit-base-int8-executorch.pte",
                    ExecuTorchImageClassificationAdapter.PROFILE_SYMMETRIC_DIRECT_RESIZE
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
        models.addAll(CompatibleModelRegistry.models());
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
