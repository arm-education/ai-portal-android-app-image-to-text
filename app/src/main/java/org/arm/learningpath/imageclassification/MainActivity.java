package org.arm.learningpath.imageclassification;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int IMPORT_MODEL_REQUEST = 100;
    private static final int OPEN_IMAGE_REQUEST = 101;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private File modelsDirectory;
    private ModelTask selectedTask;
    private ModelImporter.State activeState;
    private ModelRunner activeRunner;
    private Bitmap selectedBitmap;
    private ImageView imagePreview;
    private TextView modelStatus;
    private TextView modelExplanation;
    private TextView results;
    private EditText candidateLabels;
    private View candidateSection;
    private View quickMode;
    private View customMode;
    private ProgressBar progress;
    private Button importModel;
    private Button chooseImage;
    private Button runModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        modelsDirectory = new File(getFilesDir(), "image-models");
        imagePreview = findViewById(R.id.image_preview);
        imagePreview.setClipToOutline(true);
        modelStatus = findViewById(R.id.model_status);
        modelExplanation = findViewById(R.id.model_explanation);
        results = findViewById(R.id.results);
        candidateLabels = findViewById(R.id.candidate_labels);
        candidateSection = findViewById(R.id.candidate_section);
        quickMode = findViewById(R.id.quick_mode);
        customMode = findViewById(R.id.custom_mode);
        progress = findViewById(R.id.progress);
        importModel = findViewById(R.id.import_model);
        chooseImage = findViewById(R.id.choose_image);
        runModel = findViewById(R.id.run_model);

        quickMode.setOnClickListener(
                view -> selectMode(ModelTask.FIXED_LABEL_CLASSIFICATION)
        );
        customMode.setOnClickListener(
                view -> selectMode(ModelTask.DESCRIPTION_MATCHING)
        );
        importModel.setOnClickListener(view -> openModelPicker());
        chooseImage.setOnClickListener(view -> openImagePicker());
        runModel.setOnClickListener(view -> runInference());

        activeState = ModelImporter.activeState(modelsDirectory);
        selectedTask = activeState == null
                ? ModelTask.FIXED_LABEL_CLASSIFICATION
                : activeState.task();
        applyModeState();
        if (activeState != null) {
            results.setText(selectedTask == ModelTask.DESCRIPTION_MATCHING
                    ? R.string.clip_ready
                    : R.string.litert_ready);
        }
    }

    private void selectMode(ModelTask task) {
        if (selectedTask == task) {
            return;
        }
        closeRunners();
        selectedTask = task;
        activeState = ModelImporter.stateForTask(modelsDirectory, task);
        if (activeState != null) {
            try {
                ModelImporter.activate(modelsDirectory, activeState);
            } catch (Exception exception) {
                showError(getString(R.string.model_import_failed, exception.getMessage()));
                return;
            }
        }
        applyModeState();
        results.setText(activeState == null
                ? R.string.initial_instructions
                : task == ModelTask.DESCRIPTION_MATCHING
                ? R.string.clip_ready
                : R.string.litert_ready);
    }

    private void openModelPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                "application/octet-stream",
                "application/x-pytorch"
        });
        startActivityForResult(intent, IMPORT_MODEL_REQUEST);
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, OPEN_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }

        if (requestCode == IMPORT_MODEL_REQUEST) {
            importSelectedModel(data.getData());
        } else if (requestCode == OPEN_IMAGE_REQUEST) {
            loadSelectedImage(data.getData());
        }
    }

    private void importSelectedModel(Uri modelUri) {
        setBusy(getString(R.string.importing_model));
        executor.execute(() -> {
            try {
                closeRunners();
                ModelImporter.ImportResult imported = ModelImporter.importModel(
                        getApplicationContext(),
                        modelUri,
                        modelsDirectory
                );
                activeState = imported.state();
                selectedTask = activeState.task();
                runOnUiThread(() -> {
                    applyModeState();
                    results.setText(selectedTask == ModelTask.DESCRIPTION_MATCHING
                            ? R.string.clip_ready
                            : R.string.litert_ready);
                    setBusy(false);
                });
            } catch (Exception exception) {
                runOnUiThread(() -> showError(
                        getString(R.string.model_import_failed, exception.getMessage())
                ));
            }
        });
    }

    private void loadSelectedImage(Uri imageUri) {
        try (InputStream input = getContentResolver().openInputStream(imageUri)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                throw new IOException("Android could not decode the selected photo.");
            }
            replaceSelectedBitmap(bitmap);
            results.setText(R.string.image_ready);
            updateControls(false);
        } catch (IOException exception) {
            showError(getString(R.string.image_open_failed, exception.getMessage()));
        }
    }

    private void replaceSelectedBitmap(Bitmap bitmap) {
        if (selectedBitmap != null && selectedBitmap != bitmap && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }
        selectedBitmap = bitmap;
        imagePreview.setImageBitmap(bitmap);
    }

    private void runInference() {
        if (activeState == null || activeState.task() != selectedTask) {
            showError(getString(R.string.import_before_running));
            return;
        }
        if (selectedBitmap == null) {
            showError(getString(R.string.choose_before_running));
            return;
        }

        List<String> labels = selectedTask == ModelTask.DESCRIPTION_MATCHING
                ? parseLabels(candidateLabels.getText().toString())
                : List.of();
        setBusy(selectedTask == ModelTask.DESCRIPTION_MATCHING
                ? getString(R.string.running_clip)
                : getString(R.string.running_litert));

        executor.execute(() -> {
            try {
                String output = runActiveModel(labels);
                runOnUiThread(() -> {
                    results.setText(output);
                    setBusy(false);
                });
            } catch (Exception exception) {
                runOnUiThread(() -> showError(
                        getString(R.string.inference_failed, exception.getMessage())
                ));
            }
        });
    }

    private String runActiveModel(List<String> descriptions) throws Exception {
        if (activeRunner == null) {
            activeRunner = ModelRunnerRegistry.create(
                    getApplicationContext(),
                    ModelImporter.modelFile(modelsDirectory, activeState.task()),
                    activeState.descriptor()
            );
        }
        return activeRunner.run(selectedBitmap, descriptions);
    }

    private void applyModeState() {
        boolean customMatch = selectedTask == ModelTask.DESCRIPTION_MATCHING;
        quickMode.setBackgroundResource(customMatch
                ? R.drawable.bg_mode_unselected
                : R.drawable.bg_mode_selected);
        customMode.setBackgroundResource(customMatch
                ? R.drawable.bg_mode_selected
                : R.drawable.bg_mode_unselected);
        candidateSection.setVisibility(customMatch ? View.VISIBLE : View.GONE);
        runModel.setText(customMatch ? R.string.compare_photo : R.string.identify_photo);
        importModel.setText(customMatch
                ? R.string.import_custom_model
                : R.string.import_quick_model);

        if (activeState == null || activeState.task() != selectedTask) {
            modelStatus.setText(R.string.model_not_imported);
            modelExplanation.setText(customMatch
                    ? R.string.custom_model_missing
                    : R.string.quick_model_missing);
        } else {
            modelStatus.setText(activeState.descriptor().displayLabel());
            modelExplanation.setText(customMatch
                    ? R.string.custom_model_ready
                    : R.string.quick_model_ready);
        }
        updateControls(false);
    }

    private static List<String> parseLabels(String value) {
        Set<String> uniqueLabels = new LinkedHashSet<>();
        for (String candidate : value.split("[,\\n]")) {
            String label = candidate.trim();
            if (!label.isEmpty()) {
                uniqueLabels.add(label);
            }
        }
        return new ArrayList<>(uniqueLabels);
    }

    private void setBusy(String message) {
        progress.setVisibility(View.VISIBLE);
        results.setText(message);
        updateControls(true);
    }

    private void setBusy(boolean busy) {
        progress.setVisibility(busy ? View.VISIBLE : View.GONE);
        updateControls(busy);
    }

    private void updateControls(boolean busy) {
        quickMode.setEnabled(!busy);
        customMode.setEnabled(!busy);
        importModel.setEnabled(!busy);
        chooseImage.setEnabled(!busy);
        candidateLabels.setEnabled(!busy);
        runModel.setEnabled(!busy
                && activeState != null
                && activeState.task() == selectedTask
                && selectedBitmap != null);
    }

    private void showError(String message) {
        setBusy(false);
        results.setText(message);
    }

    private void closeRunners() {
        if (activeRunner != null) {
            activeRunner.close();
            activeRunner = null;
        }
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        closeRunners();
        if (selectedBitmap != null && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }
        super.onDestroy();
    }
}
