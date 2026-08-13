package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import java.io.File;

final class LiteRtImageClassificationAdapter implements VisionAdapter {
    static final String ID = "litert-image-classification";

    private static final AdapterDefinition DEFINITION = new AdapterDefinition(
            ID,
            R.string.quick_mode_title,
            R.string.quick_mode_description,
            R.string.import_quick_model,
            R.string.identify_photo,
            R.string.quick_model_missing,
            R.string.quick_model_ready,
            R.string.litert_ready,
            R.string.running_litert
    );

    @Override
    public AdapterDefinition definition() {
        return DEFINITION;
    }

    @Override
    public View createOptionsView(LayoutInflater inflater, ViewGroup parent) {
        return null;
    }

    @Override
    public AdapterInput collectInput(View optionsView) {
        return new EmptyAdapterInput();
    }

    @Override
    public ModelRunner createRunner(Context context, File modelFile,
                                    ModelDescriptor descriptor) throws Exception {
        if (!ID.equals(descriptor.adapterId())) {
            throw new IllegalArgumentException("The LiteRT adapter received another model type.");
        }
        LiteRtClassifier classifier = new LiteRtClassifier(
                context,
                modelFile,
                descriptor.displayLabel(),
                preprocessingProfile(descriptor)
        );
        return new ModelRunner() {
            @Override
            public String run(Bitmap bitmap, AdapterInput input) {
                if (!(input instanceof EmptyAdapterInput)) {
                    throw new IllegalArgumentException(
                            "The LiteRT classifier does not accept text input."
                    );
                }
                return classifier.classify(bitmap).formattedResults();
            }

            @Override
            public void close() {
                classifier.close();
            }
        };
    }

    private static LiteRtClassifier.PreprocessingProfile preprocessingProfile(
            ModelDescriptor descriptor) {
        return switch (descriptor.id()) {
            case "google-vit-litert" ->
                    LiteRtClassifier.PreprocessingProfile.SYMMETRIC_DIRECT_RESIZE;
            case "timm-vit-litert" ->
                    LiteRtClassifier.PreprocessingProfile.SYMMETRIC_CROP_232;
            case "deit-tiny-litert", "mobilenet-v3-small-litert", "swin-tiny-litert" ->
                    LiteRtClassifier.PreprocessingProfile.IMAGENET_CROP_256;
            default -> throw new IllegalArgumentException(
                    "No LiteRT preprocessing profile is registered for " + descriptor.id()
            );
        };
    }
}
