package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import java.io.File;

final class ExecuTorchImageClassificationAdapter implements VisionAdapter {
    static final String ID = "executorch-image-classification";
    static final String PROFILE_IMAGENET_CROP_256 = "imagenet-crop-256";
    static final String PROFILE_IMAGENET_CROP_342 = "imagenet-crop-342";
    static final String PROFILE_IMAGENET_CROP_232 = "imagenet-crop-232";
    static final String PROFILE_SYMMETRIC_DIRECT_RESIZE = "symmetric-direct-resize";

    private static final AdapterDefinition DEFINITION = new AdapterDefinition(
            ID,
            R.string.executorch_classification_mode_title,
            R.string.executorch_classification_mode_description,
            R.string.import_executorch_classifier,
            R.string.identify_photo,
            R.string.executorch_classifier_missing,
            R.string.executorch_classifier_ready,
            R.string.executorch_classification_ready,
            R.string.running_executorch_classification
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
            throw new IllegalArgumentException(
                    "The ExecuTorch classification adapter received another model type."
            );
        }
        ExecuTorchImageClassifier classifier = new ExecuTorchImageClassifier(
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
                            "The ExecuTorch classifier does not accept text input."
                    );
                }
                return classifier.classify(bitmap);
            }

            @Override
            public void close() {
                classifier.close();
            }
        };
    }

    private static ExecuTorchImageClassifier.PreprocessingProfile preprocessingProfile(
            ModelDescriptor descriptor) {
        return switch (descriptor.configurationId()) {
            case PROFILE_IMAGENET_CROP_342 ->
                    ExecuTorchImageClassifier.PreprocessingProfile.IMAGENET_CROP_342;
            case PROFILE_IMAGENET_CROP_232 ->
                    ExecuTorchImageClassifier.PreprocessingProfile.IMAGENET_CROP_232;
            case PROFILE_SYMMETRIC_DIRECT_RESIZE ->
                    ExecuTorchImageClassifier.PreprocessingProfile.SYMMETRIC_DIRECT_RESIZE;
            case PROFILE_IMAGENET_CROP_256 ->
                    ExecuTorchImageClassifier.PreprocessingProfile.IMAGENET_CROP_256;
            default -> throw new IllegalArgumentException(
                    "No ExecuTorch preprocessing profile is registered for "
                            + descriptor.configurationId()
            );
        };
    }
}
