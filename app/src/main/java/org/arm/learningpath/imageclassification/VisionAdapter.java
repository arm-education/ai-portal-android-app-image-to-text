package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import java.io.File;

interface VisionAdapter {
    AdapterDefinition definition();

    View createOptionsView(LayoutInflater inflater, ViewGroup parent);

    AdapterInput collectInput(View optionsView) throws Exception;

    ModelRunner createRunner(Context context, File modelFile, ModelDescriptor descriptor)
            throws Exception;

    default void validateModel(Context context, File modelFile, ModelDescriptor descriptor)
            throws Exception {
        try (ModelRunner ignored = createRunner(context, modelFile, descriptor)) {
        }
    }
}
