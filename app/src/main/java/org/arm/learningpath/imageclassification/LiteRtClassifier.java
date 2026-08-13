package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

final class LiteRtClassifier implements Closeable {
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STANDARD_DEVIATION = {0.229f, 0.224f, 0.225f};
    private static final float[] HALF_MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] HALF_STANDARD_DEVIATION = {0.5f, 0.5f, 0.5f};

    private final Interpreter interpreter;
    private final String modelName;
    private final PreprocessingProfile preprocessingProfile;
    private final List<String> labels;
    private final boolean channelFirst;
    private final int inputHeight;
    private final int inputWidth;

    LiteRtClassifier(Context context, File modelFile, String displayName,
                     PreprocessingProfile profile) throws Exception {
        modelName = displayName;
        preprocessingProfile = profile;
        if (preprocessingProfile == null) {
            throw new IllegalArgumentException("The classifier is missing a preprocessing profile.");
        }
        labels = readLabels(context);

        Interpreter.Options options = new Interpreter.Options()
                .setNumThreads(4)
                .setUseXNNPACK(true);
        interpreter = new Interpreter(modelFile, options);

        Tensor inputTensor = interpreter.getInputTensor(0);
        int[] inputShape = inputTensor.shape();
        if (inputTensor.dataType() != DataType.FLOAT32 || inputShape.length != 4) {
            throw new IllegalArgumentException("The sample expects one four-dimensional float32 input tensor.");
        }

        channelFirst = inputShape[1] == 3;
        boolean channelLast = inputShape[3] == 3;
        if (!channelFirst && !channelLast) {
            throw new IllegalArgumentException("The input tensor must use NCHW or NHWC RGB layout.");
        }

        inputHeight = channelFirst ? inputShape[2] : inputShape[1];
        inputWidth = channelFirst ? inputShape[3] : inputShape[2];
        validateOutputTensor(interpreter.getOutputTensor(0));
    }

    public Classification classify(Bitmap source) {
        Bitmap prepared = prepareBitmap(source);
        ByteBuffer input;
        try {
            input = createInputBuffer(prepared);
        } finally {
            if (prepared != source) {
                prepared.recycle();
            }
        }
        Tensor outputTensor = interpreter.getOutputTensor(0);
        ByteBuffer output = ByteBuffer.allocateDirect(outputTensor.numBytes())
                .order(ByteOrder.nativeOrder());

        long start = System.nanoTime();
        interpreter.run(input, output);
        long elapsedMilliseconds = (System.nanoTime() - start) / 1_000_000;

        float[] probabilities = softmax(readOutput(output, outputTensor));
        List<Prediction> predictions = topPredictions(probabilities, 5);
        return new Classification(modelName, elapsedMilliseconds, predictions);
    }

    private Bitmap prepareBitmap(Bitmap source) {
        if (preprocessingProfile.resizeMode == ResizeMode.DIRECT) {
            return Bitmap.createScaledBitmap(source, inputWidth, inputHeight, true);
        }

        float scale = preprocessingProfile.shortEdge / (float) Math.min(
                source.getWidth(),
                source.getHeight()
        );
        int scaledWidth = Math.max(inputWidth, Math.round(source.getWidth() * scale));
        int scaledHeight = Math.max(inputHeight, Math.round(source.getHeight() * scale));
        Bitmap scaled = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true);
        int left = Math.max(0, (scaledWidth - inputWidth) / 2);
        int top = Math.max(0, (scaledHeight - inputHeight) / 2);
        Bitmap cropped = Bitmap.createBitmap(scaled, left, top, inputWidth, inputHeight);
        if (cropped != scaled && scaled != source) {
            scaled.recycle();
        }
        return cropped;
    }

    private ByteBuffer createInputBuffer(Bitmap bitmap) {
        ByteBuffer input = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        int[] pixels = new int[inputWidth * inputHeight];
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);

        if (channelFirst) {
            for (int channel = 0; channel < 3; channel++) {
                for (int pixel : pixels) {
                    input.putFloat(normalize(channelValue(pixel, channel), channel));
                }
            }
        } else {
            for (int pixel : pixels) {
                for (int channel = 0; channel < 3; channel++) {
                    input.putFloat(normalize(channelValue(pixel, channel), channel));
                }
            }
        }
        input.rewind();
        return input;
    }

    private static int channelValue(int pixel, int channel) {
        return switch (channel) {
            case 0 -> (pixel >> 16) & 0xff;
            case 1 -> (pixel >> 8) & 0xff;
            default -> pixel & 0xff;
        };
    }

    private float normalize(int value, int channel) {
        float scaled = value / 255.0f;
        return (scaled - preprocessingProfile.mean[channel])
                / preprocessingProfile.standardDeviation[channel];
    }

    private static void validateOutputTensor(Tensor tensor) {
        DataType dataType = tensor.dataType();
        if (dataType != DataType.FLOAT32
                && dataType != DataType.INT8
                && dataType != DataType.UINT8) {
            throw new IllegalArgumentException("Unsupported output type: " + dataType);
        }
        if (tensor.numElements() != 1000) {
            throw new IllegalArgumentException(
                    "The launch classifier expects 1,000 ImageNet output scores."
            );
        }
    }

    private enum ResizeMode {
        DIRECT,
        SHORT_EDGE_CENTER_CROP
    }

    enum PreprocessingProfile {
        IMAGENET_CROP_256(
                ResizeMode.SHORT_EDGE_CENTER_CROP,
                256,
                IMAGENET_MEAN,
                IMAGENET_STANDARD_DEVIATION
        ),
        SYMMETRIC_DIRECT_RESIZE(
                ResizeMode.DIRECT,
                0,
                HALF_MEAN,
                HALF_STANDARD_DEVIATION
        ),
        SYMMETRIC_CROP_232(
                ResizeMode.SHORT_EDGE_CENTER_CROP,
                232,
                HALF_MEAN,
                HALF_STANDARD_DEVIATION
        );

        private final ResizeMode resizeMode;
        private final int shortEdge;
        private final float[] mean;
        private final float[] standardDeviation;

        PreprocessingProfile(ResizeMode resizeMode, int shortEdge,
                             float[] mean, float[] standardDeviation) {
            this.resizeMode = resizeMode;
            this.shortEdge = shortEdge;
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }
    }

    private static float[] readOutput(ByteBuffer output, Tensor tensor) {
        output.rewind();
        float[] scores = new float[tensor.numElements()];
        DataType dataType = tensor.dataType();

        if (dataType == DataType.FLOAT32) {
            output.asFloatBuffer().get(scores);
            return scores;
        }

        Tensor.QuantizationParams quantization = tensor.quantizationParams();
        float scale = quantization.getScale();
        int zeroPoint = quantization.getZeroPoint();
        for (int index = 0; index < scores.length; index++) {
            int quantizedValue;
            if (dataType == DataType.INT8) {
                quantizedValue = output.get();
            } else if (dataType == DataType.UINT8) {
                quantizedValue = output.get() & 0xff;
            } else {
                throw new IllegalArgumentException("Unsupported output type: " + dataType);
            }
            scores[index] = (quantizedValue - zeroPoint) * scale;
        }
        return scores;
    }

    private static float[] softmax(float[] logits) {
        float maximum = Float.NEGATIVE_INFINITY;
        for (float value : logits) {
            maximum = Math.max(maximum, value);
        }

        double sum = 0.0;
        float[] probabilities = new float[logits.length];
        for (int index = 0; index < logits.length; index++) {
            probabilities[index] = (float) Math.exp(logits[index] - maximum);
            sum += probabilities[index];
        }
        for (int index = 0; index < probabilities.length; index++) {
            probabilities[index] /= (float) sum;
        }
        return probabilities;
    }

    private List<Prediction> topPredictions(float[] probabilities, int count) {
        List<Prediction> predictions = new ArrayList<>(probabilities.length);
        for (int index = 0; index < probabilities.length; index++) {
            String label = index < labels.size() ? labels.get(index) : "class " + index;
            predictions.add(new Prediction(label, probabilities[index]));
        }
        predictions.sort(Comparator.comparing(Prediction::probability).reversed());
        return predictions.subList(0, Math.min(count, predictions.size()));
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
        List<String> result = new ArrayList<>();
        if (json instanceof JSONArray array) {
            for (int index = 0; index < array.length(); index++) {
                result.add(array.getString(index));
            }
            return result;
        }

        JSONObject object = (JSONObject) json;
        for (int index = 0; index < object.length(); index++) {
            result.add(object.getString(Integer.toString(index)));
        }
        return result;
    }

    @Override
    public void close() {
        interpreter.close();
    }

    public record Prediction(String label, float probability) {
    }

    public record Classification(String modelName, long elapsedMilliseconds,
                                 List<Prediction> predictions) {
        public String formattedResults() {
            StringBuilder text = new StringBuilder();
            text.append("Processing time: ").append(elapsedMilliseconds).append(" ms\n\n");
            for (int index = 0; index < predictions.size(); index++) {
                Prediction prediction = predictions.get(index);
                text.append(String.format(
                        Locale.US,
                        "%d. %5.1f%%  %s%n",
                        index + 1,
                        prediction.probability() * 100.0f,
                        prediction.label()
                ));
            }
            return text.toString();
        }
    }
}
