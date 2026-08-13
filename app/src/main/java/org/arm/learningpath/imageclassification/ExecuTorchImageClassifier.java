package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.pytorch.executorch.DType;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.MethodMetadata;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

final class ExecuTorchImageClassifier implements Closeable {
    private static final int CLASS_COUNT = 1000;
    private static final int TOP_K = 5;
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STANDARD_DEVIATION = {0.229f, 0.224f, 0.225f};
    private static final float[] HALF_MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] HALF_STANDARD_DEVIATION = {0.5f, 0.5f, 0.5f};

    private final Module module;
    private final PreprocessingProfile preprocessingProfile;
    private final List<String> labels;

    ExecuTorchImageClassifier(Context context, File modelFile,
                              PreprocessingProfile profile) throws Exception {
        if (!modelFile.isFile() || modelFile.length() == 0) {
            throw new IllegalArgumentException("The imported ExecuTorch model is unavailable.");
        }
        preprocessingProfile = profile;
        labels = readLabels(context);

        Module loadedModule = null;
        try {
            loadedModule = Module.load(modelFile.getAbsolutePath(), Module.LOAD_MODE_MMAP);
            validateModule(loadedModule);
            module = loadedModule;
        } catch (Exception exception) {
            if (loadedModule != null) {
                loadedModule.close();
            }
            throw exception;
        }
    }

    String classify(Bitmap source) {
        if (source == null || source.isRecycled()
                || source.getWidth() < 1 || source.getHeight() < 1) {
            throw new IllegalArgumentException("Choose a valid photo before running inference.");
        }

        float[] inputData = preprocess(source);
        int imageSize = preprocessingProfile.imageSize;
        long[] inputShape = {1, 3, imageSize, imageSize};
        Tensor inputTensor = Tensor.fromBlob(inputData, inputShape);

        long started = System.nanoTime();
        EValue[] outputs = module.execute("forward", EValue.from(inputTensor));
        long elapsedMilliseconds = Math.round((System.nanoTime() - started) / 1_000_000.0);
        float[] logits = validateOutput(outputs);
        return formatResults(logits, elapsedMilliseconds);
    }

    private static void validateModule(Module module) {
        Set<String> methods = new HashSet<>(Arrays.asList(module.getMethods()));
        if (!methods.contains("forward")) {
            throw new IllegalArgumentException("The model is missing the forward method.");
        }
        module.loadMethod("forward");
        MethodMetadata metadata = module.getMethodMetadata("forward");
        Set<String> backends = new HashSet<>(Arrays.asList(metadata.getBackends()));
        if (!backends.contains("XnnpackBackend")) {
            throw new IllegalArgumentException(
                    "The model forward method does not declare the XNNPACK backend."
            );
        }
    }

    private static float[] validateOutput(EValue[] outputs) {
        if (outputs == null || outputs.length != 1 || !outputs[0].isTensor()) {
            throw new IllegalStateException("The forward method must return one tensor.");
        }
        Tensor output = outputs[0].toTensor();
        if (output.dtype() != DType.FLOAT) {
            throw new IllegalStateException("The output tensor must contain float32 logits.");
        }
        if (!Arrays.equals(output.shape(), new long[]{1, CLASS_COUNT})) {
            throw new IllegalStateException(
                    "The output tensor must have shape [1, 1000], but was "
                            + Arrays.toString(output.shape())
            );
        }
        float[] logits = output.getDataAsFloatArray();
        for (float logit : logits) {
            if (!Float.isFinite(logit)) {
                throw new IllegalStateException("The output contains a non-finite logit.");
            }
        }
        return logits;
    }

    private float[] preprocess(Bitmap source) {
        int imageSize = preprocessingProfile.imageSize;
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();
        double scaleX;
        double scaleY;
        double cropLeft;
        double cropTop;

        if (preprocessingProfile.resizeMode == ResizeMode.DIRECT) {
            scaleX = (double) imageSize / sourceWidth;
            scaleY = (double) imageSize / sourceHeight;
            cropLeft = 0.0;
            cropTop = 0.0;
        } else {
            double scale = (double) preprocessingProfile.shortEdge
                    / Math.min(sourceWidth, sourceHeight);
            int resizedWidth = Math.max(imageSize, (int) Math.round(sourceWidth * scale));
            int resizedHeight = Math.max(imageSize, (int) Math.round(sourceHeight * scale));
            scaleX = (double) resizedWidth / sourceWidth;
            scaleY = (double) resizedHeight / sourceHeight;
            cropLeft = (resizedWidth - imageSize) / 2.0;
            cropTop = (resizedHeight - imageSize) / 2.0;
        }

        int[] sourcePixels = new int[sourceWidth * sourceHeight];
        source.getPixels(sourcePixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight);
        float[] tensorData = new float[3 * imageSize * imageSize];
        int planeSize = imageSize * imageSize;
        for (int y = 0; y < imageSize; y++) {
            double sourceY = (cropTop + y + 0.5) / scaleY - 0.5;
            for (int x = 0; x < imageSize; x++) {
                double sourceX = (cropLeft + x + 0.5) / scaleX - 0.5;
                int color = bicubicPixel(
                        sourcePixels,
                        sourceWidth,
                        sourceHeight,
                        sourceX,
                        sourceY
                );
                int offset = y * imageSize + x;
                tensorData[offset] = normalize(Color.red(color), 0);
                tensorData[planeSize + offset] = normalize(Color.green(color), 1);
                tensorData[2 * planeSize + offset] = normalize(Color.blue(color), 2);
            }
        }
        return tensorData;
    }

    private float normalize(int channelValue, int channel) {
        float unitValue = channelValue / 255.0f;
        return (unitValue - preprocessingProfile.mean[channel])
                / preprocessingProfile.standardDeviation[channel];
    }

    private static int bicubicPixel(int[] pixels, int width, int height, double x, double y) {
        int baseX = (int) Math.floor(x);
        int baseY = (int) Math.floor(y);
        double red = 0.0;
        double green = 0.0;
        double blue = 0.0;
        double weightTotal = 0.0;

        for (int offsetY = -1; offsetY <= 2; offsetY++) {
            int sampleY = clamp(baseY + offsetY, 0, height - 1);
            double weightY = cubicWeight(y - (baseY + offsetY));
            for (int offsetX = -1; offsetX <= 2; offsetX++) {
                int sampleX = clamp(baseX + offsetX, 0, width - 1);
                double weight = weightY * cubicWeight(x - (baseX + offsetX));
                int color = pixels[sampleY * width + sampleX];
                red += Color.red(color) * weight;
                green += Color.green(color) * weight;
                blue += Color.blue(color) * weight;
                weightTotal += weight;
            }
        }

        if (Math.abs(weightTotal) > 1.0e-12) {
            red /= weightTotal;
            green /= weightTotal;
            blue /= weightTotal;
        }
        return Color.rgb(
                clamp((int) Math.round(red), 0, 255),
                clamp((int) Math.round(green), 0, 255),
                clamp((int) Math.round(blue), 0, 255)
        );
    }

    private static double cubicWeight(double value) {
        double distance = Math.abs(value);
        double coefficient = -0.75;
        if (distance <= 1.0) {
            return (coefficient + 2.0) * distance * distance * distance
                    - (coefficient + 3.0) * distance * distance + 1.0;
        }
        if (distance < 2.0) {
            return coefficient * distance * distance * distance
                    - 5.0 * coefficient * distance * distance
                    + 8.0 * coefficient * distance - 4.0 * coefficient;
        }
        return 0.0;
    }

    private String formatResults(float[] logits, long elapsedMilliseconds) {
        float maximumLogit = -Float.MAX_VALUE;
        for (float logit : logits) {
            maximumLogit = Math.max(maximumLogit, logit);
        }

        double probabilityTotal = 0.0;
        double[] probabilities = new double[CLASS_COUNT];
        List<Integer> indices = new ArrayList<>(CLASS_COUNT);
        for (int index = 0; index < CLASS_COUNT; index++) {
            probabilities[index] = Math.exp(logits[index] - maximumLogit);
            probabilityTotal += probabilities[index];
            indices.add(index);
        }
        if (!Double.isFinite(probabilityTotal) || probabilityTotal <= 0.0) {
            throw new IllegalStateException("The logits could not be converted to probabilities.");
        }
        for (int index = 0; index < CLASS_COUNT; index++) {
            probabilities[index] /= probabilityTotal;
        }
        indices.sort(Comparator.comparingDouble(
                (Integer index) -> probabilities[index]
        ).reversed());

        StringBuilder output = new StringBuilder();
        output.append("Processing time: ").append(elapsedMilliseconds).append(" ms\n\n");
        for (int rank = 0; rank < TOP_K; rank++) {
            int index = indices.get(rank);
            output.append(String.format(
                    Locale.US,
                    "%d. %5.1f%%  %s%n",
                    rank + 1,
                    probabilities[index] * 100.0,
                    labels.get(index)
            ));
        }
        return output.toString();
    }

    private static List<String> readLabels(Context context) throws Exception {
        StringBuilder text = new StringBuilder();
        try (InputStream input = context.getAssets().open("imagenet_classes.json");
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(input, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                text.append(line);
            }
        }
        Object json = new JSONTokener(text.toString()).nextValue();
        List<String> result = new ArrayList<>(CLASS_COUNT);
        if (json instanceof JSONArray array) {
            for (int index = 0; index < array.length(); index++) {
                result.add(array.getString(index));
            }
        } else {
            JSONObject object = (JSONObject) json;
            for (int index = 0; index < object.length(); index++) {
                result.add(object.getString(Integer.toString(index)));
            }
        }
        if (result.size() != CLASS_COUNT) {
            throw new IllegalArgumentException("The ImageNet label file must contain 1,000 labels.");
        }
        return List.copyOf(result);
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    @Override
    public void close() {
        module.close();
    }

    private enum ResizeMode {
        DIRECT,
        SHORT_EDGE_CENTER_CROP
    }

    enum PreprocessingProfile {
        IMAGENET_CROP_256(
                224,
                ResizeMode.SHORT_EDGE_CENTER_CROP,
                256,
                IMAGENET_MEAN,
                IMAGENET_STANDARD_DEVIATION
        ),
        IMAGENET_CROP_342(
                299,
                ResizeMode.SHORT_EDGE_CENTER_CROP,
                342,
                IMAGENET_MEAN,
                IMAGENET_STANDARD_DEVIATION
        ),
        IMAGENET_CROP_232(
                224,
                ResizeMode.SHORT_EDGE_CENTER_CROP,
                232,
                IMAGENET_MEAN,
                IMAGENET_STANDARD_DEVIATION
        ),
        SYMMETRIC_DIRECT_RESIZE(
                224,
                ResizeMode.DIRECT,
                0,
                HALF_MEAN,
                HALF_STANDARD_DEVIATION
        );

        private final int imageSize;
        private final ResizeMode resizeMode;
        private final int shortEdge;
        private final float[] mean;
        private final float[] standardDeviation;

        PreprocessingProfile(int imageSize, ResizeMode resizeMode, int shortEdge,
                             float[] mean, float[] standardDeviation) {
            this.imageSize = imageSize;
            this.resizeMode = resizeMode;
            this.shortEdge = shortEdge;
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }
    }
}
