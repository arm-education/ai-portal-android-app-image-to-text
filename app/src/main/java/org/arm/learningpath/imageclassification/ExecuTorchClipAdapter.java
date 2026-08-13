package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

final class ExecuTorchClipAdapter implements VisionAdapter {
    static final String ID = "executorch-clip-matching";

    private static final AdapterDefinition DEFINITION = new AdapterDefinition(
            ID,
            R.string.custom_mode_title,
            R.string.custom_mode_description,
            R.string.import_custom_model,
            R.string.compare_photo,
            R.string.custom_model_missing,
            R.string.custom_model_ready,
            R.string.clip_ready,
            R.string.running_clip
    );

    @Override
    public AdapterDefinition definition() {
        return DEFINITION;
    }

    @Override
    public View createOptionsView(LayoutInflater inflater, ViewGroup parent) {
        return inflater.inflate(R.layout.adapter_clip_options, parent, false);
    }

    @Override
    public AdapterInput collectInput(View optionsView) {
        EditText candidateLabels = optionsView.findViewById(R.id.candidate_labels);
        List<String> descriptions = parseDescriptions(candidateLabels.getText().toString());
        if (descriptions.size() < 2) {
            throw new IllegalArgumentException("Enter at least two descriptions to compare.");
        }
        return new DescriptionAdapterInput(descriptions);
    }

    @Override
    public ModelRunner createRunner(Context context, File modelFile,
                                    ModelDescriptor descriptor) throws Exception {
        if (!ID.equals(descriptor.adapterId())) {
            throw new IllegalArgumentException("The CLIP adapter received another model type.");
        }
        ClipModel clipModel = new ClipModel(context, modelFile);
        return new ModelRunner() {
            @Override
            public String run(Bitmap bitmap, AdapterInput input) {
                if (!(input instanceof DescriptionAdapterInput descriptionInput)) {
                    throw new IllegalArgumentException("The CLIP adapter requires descriptions.");
                }
                ClipModel.MatchResult result = clipModel.match(
                        bitmap,
                        descriptionInput.descriptions()
                );
                StringBuilder output = new StringBuilder();
                output.append(String.format(
                        Locale.US,
                        "Processing time: %,d ms%n%n",
                        result.elapsedMilliseconds()
                ));
                int rank = 1;
                for (ClipModel.ScoredLabel label : result.labels()) {
                    output.append(label.display(rank++)).append('\n');
                }
                return output.toString();
            }

            @Override
            public void close() {
                clipModel.close();
            }
        };
    }

    private static List<String> parseDescriptions(String value) {
        Set<String> uniqueDescriptions = new LinkedHashSet<>();
        for (String candidate : value.split("[,\\n]")) {
            String description = candidate.trim();
            if (!description.isEmpty()) {
                uniqueDescriptions.add(description);
            }
        }
        return new ArrayList<>(uniqueDescriptions);
    }
}
