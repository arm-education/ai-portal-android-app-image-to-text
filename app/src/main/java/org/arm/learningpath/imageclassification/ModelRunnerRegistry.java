package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.File;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

final class ModelRunnerRegistry {
    private static final Map<ModelDescriptor.ModelRuntime, RunnerFactory> FACTORIES =
            new EnumMap<>(ModelDescriptor.ModelRuntime.class);

    static {
        FACTORIES.put(ModelDescriptor.ModelRuntime.LITERT, ModelRunnerRegistry::createLiteRt);
        FACTORIES.put(ModelDescriptor.ModelRuntime.EXECUTORCH, ModelRunnerRegistry::createClip);
    }

    private ModelRunnerRegistry() {
    }

    static ModelRunner create(Context context, File modelFile, ModelDescriptor descriptor)
            throws Exception {
        RunnerFactory factory = FACTORIES.get(descriptor.runtime());
        if (factory == null) {
            throw new IllegalArgumentException(
                    "No runner is registered for " + descriptor.runtime().displayName()
            );
        }
        return factory.create(context, modelFile, descriptor);
    }

    static void validate(Context context, File modelFile, ModelDescriptor descriptor)
            throws Exception {
        try (ModelRunner ignored = create(context, modelFile, descriptor)) {
        }
    }

    private static ModelRunner createLiteRt(Context context, File modelFile,
                                             ModelDescriptor descriptor) throws Exception {
        if (descriptor.task() != ModelTask.FIXED_LABEL_CLASSIFICATION) {
            throw new IllegalArgumentException("The LiteRT runner expects a classifier.");
        }
        LiteRtClassifier classifier = new LiteRtClassifier(context, modelFile, descriptor);
        return new ModelRunner() {
            @Override
            public String run(Bitmap bitmap, List<String> descriptions) {
                return classifier.classify(bitmap).formattedResults();
            }

            @Override
            public void close() {
                classifier.close();
            }
        };
    }

    private static ModelRunner createClip(Context context, File modelFile,
                                          ModelDescriptor descriptor) throws Exception {
        if (descriptor.task() != ModelTask.DESCRIPTION_MATCHING) {
            throw new IllegalArgumentException("The ExecuTorch runner expects CLIP matching.");
        }
        ClipModel clipModel = new ClipModel(context, modelFile);
        return new ModelRunner() {
            @Override
            public String run(Bitmap bitmap, List<String> descriptions) {
                ClipModel.MatchResult result = clipModel.match(bitmap, descriptions);
                StringBuilder output = new StringBuilder();
                int rank = 1;
                for (ClipModel.ScoredLabel label : result.labels()) {
                    output.append(label.display(rank++)).append('\n');
                }
                output.append(String.format(
                        Locale.US,
                        "\nAnalysis time: %,d ms",
                        result.elapsedMilliseconds()
                ));
                return output.toString();
            }

            @Override
            public void close() {
                clipModel.close();
            }
        };
    }

    private interface RunnerFactory {
        ModelRunner create(Context context, File modelFile, ModelDescriptor descriptor)
                throws Exception;
    }
}
