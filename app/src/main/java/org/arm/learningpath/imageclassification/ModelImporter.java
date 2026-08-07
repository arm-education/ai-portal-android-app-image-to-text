package org.arm.learningpath.imageclassification;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

final class ModelImporter {
    private static final long MAXIMUM_MODEL_BYTES = 700L * 1024L * 1024L;

    private ModelImporter() {
    }

    static ImportResult importModel(Context context, Uri source, File modelsDirectory)
            throws Exception {
        ContentResolver resolver = context.getContentResolver();
        String fileName = displayName(resolver, source);
        if (fileName == null) {
            throw new IOException("Android could not determine the selected file name.");
        }

        ModelDescriptor descriptor = ModelRegistry.forFileName(fileName);
        if (descriptor == null) {
            throw new IOException(
                    "This launch app does not recognize " + fileName
                            + ". Keep the downloaded filename unchanged and select one of: "
                            + ModelRegistry.supportedFiles()
            );
        }

        File destination = modelFile(modelsDirectory, descriptor.task());
        File parent = destination.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("Unable to create app model storage.");
        }

        File temporaryFile = new File(parent, destination.getName() + ".importing");
        if (temporaryFile.exists() && !temporaryFile.delete()) {
            throw new IOException("Unable to replace the temporary model file.");
        }
        try (InputStream input = resolver.openInputStream(source)) {
            if (input == null) {
                throw new IOException("Android could not open the selected model file.");
            }
            copy(input, temporaryFile);
            ModelRunnerRegistry.validate(context, temporaryFile, descriptor);
        } catch (Exception exception) {
            temporaryFile.delete();
            throw exception;
        }

        if (destination.exists() && !destination.delete()) {
            temporaryFile.delete();
            throw new IOException("Unable to replace the installed model.");
        }
        if (!temporaryFile.renameTo(destination)) {
            temporaryFile.delete();
            throw new IOException("Unable to activate the imported model.");
        }

        writeDescriptorId(parent, descriptor.id());
        State state = new State(descriptor);
        writeState(modelsDirectory, state);
        return new ImportResult(state);
    }

    static State activeState(File modelsDirectory) {
        File stateFile = new File(modelsDirectory, "active-model.json");
        if (!stateFile.isFile()) {
            return null;
        }
        try {
            JSONObject state = new JSONObject(readText(stateFile));
            boolean legacyState = !state.has("descriptorId");
            ModelDescriptor descriptor = legacyState
                    ? ModelRegistry.forStoredName(state.optString("modelName"))
                    : ModelRegistry.forId(state.getString("descriptorId"));
            if (descriptor == null || !isInstalled(modelsDirectory, descriptor.task())) {
                return null;
            }
            State result = new State(descriptor);
            if (legacyState) {
                writeDescriptorId(
                        modelFile(modelsDirectory, descriptor.task()).getParentFile(),
                        descriptor.id()
                );
                writeState(modelsDirectory, result);
            }
            return result;
        } catch (Exception ignored) {
            return null;
        }
    }

    static State stateForTask(File modelsDirectory, ModelTask task) {
        if (!isInstalled(modelsDirectory, task)) {
            return null;
        }
        File descriptorFile = new File(
                modelFile(modelsDirectory, task).getParentFile(),
                "descriptor-id.txt"
        );
        try {
            ModelDescriptor descriptor = ModelRegistry.forId(readText(descriptorFile).trim());
            return descriptor != null && descriptor.task() == task
                    ? new State(descriptor)
                    : null;
        } catch (Exception ignored) {
            try {
                File modelDirectory = modelFile(modelsDirectory, task).getParentFile();
                ModelDescriptor descriptor = ModelRegistry.forStoredName(readText(
                        new File(modelDirectory, "display-name.txt")
                ));
                if (descriptor == null || descriptor.task() != task) {
                    return null;
                }
                writeDescriptorId(modelDirectory, descriptor.id());
                return new State(descriptor);
            } catch (Exception legacyImportError) {
                return null;
            }
        }
    }

    static void activate(File modelsDirectory, State state) throws Exception {
        if (!isInstalled(modelsDirectory, state.task())) {
            throw new IOException("The selected model mode is not installed.");
        }
        writeState(modelsDirectory, state);
    }

    static File modelFile(File modelsDirectory, ModelTask task) {
        return task == ModelTask.DESCRIPTION_MATCHING
                ? new File(new File(modelsDirectory, "clip"), "model.pte")
                : new File(new File(modelsDirectory, "litert"), "model.tflite");
    }

    private static boolean isInstalled(File modelsDirectory, ModelTask task) {
        return modelFile(modelsDirectory, task).isFile();
    }

    private static void writeState(File modelsDirectory, State state) throws Exception {
        if (!modelsDirectory.isDirectory() && !modelsDirectory.mkdirs()) {
            throw new IOException("Unable to create app model storage.");
        }
        JSONObject value = new JSONObject();
        value.put("descriptorId", state.descriptor().id());
        try (FileOutputStream output = new FileOutputStream(
                new File(modelsDirectory, "active-model.json"))) {
            output.write(value.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void writeDescriptorId(File modelDirectory, String descriptorId)
            throws IOException {
        try (FileOutputStream output = new FileOutputStream(
                new File(modelDirectory, "descriptor-id.txt"))) {
            output.write(descriptorId.getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void copy(InputStream source, File destination) throws IOException {
        byte[] buffer = new byte[1024 * 1024];
        long total = 0;
        try (BufferedInputStream input = new BufferedInputStream(source, buffer.length);
             BufferedOutputStream output = new BufferedOutputStream(
                     new FileOutputStream(destination), buffer.length)) {
            int count;
            while ((count = input.read(buffer)) != -1) {
                total += count;
                if (total > MAXIMUM_MODEL_BYTES) {
                    throw new IOException("The selected model is larger than 700 MB.");
                }
                output.write(buffer, 0, count);
            }
        }
    }

    private static String displayName(ContentResolver resolver, Uri source) {
        try (Cursor cursor = resolver.query(
                source,
                new String[]{OpenableColumns.DISPLAY_NAME},
                null,
                null,
                null
        )) {
            if (cursor != null && cursor.moveToFirst()) {
                int nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (nameColumn >= 0) {
                    return cursor.getString(nameColumn);
                }
            }
        }
        return null;
    }

    private static String readText(File file) throws IOException {
        try (InputStream input = new FileInputStream(file)) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    record State(ModelDescriptor descriptor) {
        ModelTask task() {
            return descriptor.task();
        }
    }

    record ImportResult(State state) {
    }
}
