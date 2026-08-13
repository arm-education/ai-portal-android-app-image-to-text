package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.Closeable;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

final class ClipModel implements Closeable {
    private static final int IMAGE_SIZE = 224;
    private static final int EMBEDDING_SIZE = 512;
    private static final float LOGIT_SCALE = 100.0f;
    private static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] STANDARD_DEVIATION = {0.26862954f, 0.26130258f, 0.27577711f};

    private final Module module;
    private final ClipTokenizer tokenizer;
    private final Map<String, float[]> textEmbeddingCache = new HashMap<>();

    ClipModel(Context context, File modelFile) throws Exception {
        tokenizer = new ClipTokenizer(context);
        module = Module.load(modelFile.getAbsolutePath(), Module.LOAD_MODE_MMAP);
        Set<String> methods = new HashSet<>(Arrays.asList(module.getMethods()));
        for (String required : new String[]{"encode_image", "encode_text"}) {
            if (!methods.contains(required)) {
                module.close();
                throw new IllegalArgumentException("The model is missing method: " + required);
            }
        }
        module.loadMethod("encode_image");
        module.loadMethod("encode_text");
    }

    MatchResult match(Bitmap bitmap, List<String> labels) {
        if (labels.size() < 2) {
            throw new IllegalArgumentException("Enter at least two candidate descriptions.");
        }
        if (labels.size() > 20) {
            throw new IllegalArgumentException("Use no more than 20 candidate descriptions.");
        }

        long started = System.nanoTime();
        float[] imageEmbedding = encodeImage(bitmap);
        List<ScoredLabel> scoredLabels = new ArrayList<>();
        float maximumLogit = -Float.MAX_VALUE;

        for (String label : labels) {
            float[] textEmbedding = textEmbeddingCache.computeIfAbsent(
                    label,
                    value -> encodeText("a photo of " + value)
            );
            float logit = dotProduct(imageEmbedding, textEmbedding) * LOGIT_SCALE;
            maximumLogit = Math.max(maximumLogit, logit);
            scoredLabels.add(new ScoredLabel(label, logit, 0.0f));
        }

        double probabilityTotal = 0.0;
        for (ScoredLabel scoredLabel : scoredLabels) {
            probabilityTotal += Math.exp(scoredLabel.logit() - maximumLogit);
        }

        List<ScoredLabel> probabilities = new ArrayList<>(scoredLabels.size());
        for (ScoredLabel scoredLabel : scoredLabels) {
            float probability = (float) (
                    Math.exp(scoredLabel.logit() - maximumLogit) / probabilityTotal
            );
            probabilities.add(new ScoredLabel(
                    scoredLabel.label(),
                    scoredLabel.logit(),
                    probability
            ));
        }
        probabilities.sort(Comparator.comparing(ScoredLabel::probability).reversed());

        long elapsedMilliseconds = Math.round((System.nanoTime() - started) / 1_000_000.0);
        return new MatchResult(probabilities, elapsedMilliseconds);
    }

    private float[] encodeImage(Bitmap source) {
        Bitmap prepared = resizeAndCenterCrop(source);
        float[] tensorData = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
        int planeSize = IMAGE_SIZE * IMAGE_SIZE;

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int color = prepared.getPixel(x, y);
                int offset = y * IMAGE_SIZE + x;
                tensorData[offset] = normalize(Color.red(color), 0);
                tensorData[planeSize + offset] = normalize(Color.green(color), 1);
                tensorData[2 * planeSize + offset] = normalize(Color.blue(color), 2);
            }
        }
        if (prepared != source) {
            prepared.recycle();
        }

        Tensor imageTensor = Tensor.fromBlob(
                tensorData,
                new long[]{1, 3, IMAGE_SIZE, IMAGE_SIZE}
        );
        float[] embedding = module.execute(
                "encode_image",
                EValue.from(imageTensor)
        )[0].toTensor().getDataAsFloatArray();
        return normalizeEmbedding(embedding);
    }

    private float[] encodeText(String prompt) {
        ClipTokenizer.TokenizedPrompt tokens = tokenizer.tokenize(prompt);
        Tensor inputIds = Tensor.fromBlob(
                tokens.inputIds(),
                new long[]{1, ClipTokenizer.CONTEXT_LENGTH}
        );
        Tensor attentionMask = Tensor.fromBlob(
                tokens.attentionMask(),
                new long[]{1, ClipTokenizer.CONTEXT_LENGTH}
        );
        float[] embedding = module.execute(
                "encode_text",
                EValue.from(inputIds),
                EValue.from(attentionMask)
        )[0].toTensor().getDataAsFloatArray();
        return normalizeEmbedding(embedding);
    }

    private static Bitmap resizeAndCenterCrop(Bitmap source) {
        int width = source.getWidth();
        int height = source.getHeight();
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("The selected image has invalid dimensions.");
        }

        float scale = IMAGE_SIZE / (float) Math.min(width, height);
        int scaledWidth = Math.max(IMAGE_SIZE, Math.round(width * scale));
        int scaledHeight = Math.max(IMAGE_SIZE, Math.round(height * scale));
        Bitmap scaled = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true);
        int left = Math.max(0, (scaledWidth - IMAGE_SIZE) / 2);
        int top = Math.max(0, (scaledHeight - IMAGE_SIZE) / 2);
        Bitmap cropped = Bitmap.createBitmap(scaled, left, top, IMAGE_SIZE, IMAGE_SIZE);
        if (cropped != scaled) {
            scaled.recycle();
        }
        return cropped;
    }

    private static float normalize(int channelValue, int channel) {
        float value = channelValue / 255.0f;
        return (value - MEAN[channel]) / STANDARD_DEVIATION[channel];
    }

    private static float[] normalizeEmbedding(float[] embedding) {
        if (embedding.length != EMBEDDING_SIZE) {
            throw new IllegalStateException(
                    "Expected a 512-value CLIP embedding but received " + embedding.length
            );
        }
        double squaredTotal = 0.0;
        for (float value : embedding) {
            squaredTotal += value * value;
        }
        float length = (float) Math.sqrt(squaredTotal);
        if (length == 0.0f) {
            throw new IllegalStateException("The model returned an empty embedding.");
        }
        for (int index = 0; index < embedding.length; index++) {
            embedding[index] /= length;
        }
        return embedding;
    }

    private static float dotProduct(float[] left, float[] right) {
        float total = 0.0f;
        for (int index = 0; index < left.length; index++) {
            total += left[index] * right[index];
        }
        return total;
    }

    @Override
    public void close() {
        module.close();
        textEmbeddingCache.clear();
    }

    record ScoredLabel(String label, float logit, float probability) {
        String display(int rank) {
            return String.format(
                    Locale.US,
                    "%d. %6.2f%%  %s",
                    rank,
                    probability * 100.0f,
                    label
            );
        }
    }

    record MatchResult(List<ScoredLabel> labels, long elapsedMilliseconds) {
    }
}
