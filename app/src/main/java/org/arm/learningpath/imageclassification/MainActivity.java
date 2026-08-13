package org.arm.learningpath.imageclassification;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int IMPORT_MODEL_REQUEST = 100;
    private static final int OPEN_IMAGE_REQUEST = 101;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Map<String, View> modeCards = new LinkedHashMap<>();
    private File modelsDirectory;
    private List<VisionAdapter> adapters;
    private VisionAdapter selectedAdapter;
    private View selectedOptionsView;
    private ModelImporter.State activeState;
    private ModelRunner activeRunner;
    private Bitmap selectedBitmap;
    private ImageView imagePreview;
    private TextView modelStatus;
    private TextView modelExplanation;
    private TextView results;
    private LinearLayout modeSelector;
    private ViewGroup adapterOptions;
    private ProgressBar progress;
    private Button importModel;
    private Button chooseImage;
    private Button runModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        modelsDirectory = new File(getFilesDir(), "image-models");
        adapters = AdapterRegistry.all();
        if (adapters.isEmpty()) {
            throw new IllegalStateException("The application does not contain any vision adapters.");
        }

        imagePreview = findViewById(R.id.image_preview);
        imagePreview.setClipToOutline(true);
        modelStatus = findViewById(R.id.model_status);
        modelExplanation = findViewById(R.id.model_explanation);
        results = findViewById(R.id.results);
        modeSelector = findViewById(R.id.mode_selector);
        modeSelector.setMinimumWidth(
                getResources().getDisplayMetrics().widthPixels - dpToPixels(36)
        );
        adapterOptions = findViewById(R.id.adapter_options);
        progress = findViewById(R.id.progress);
        importModel = findViewById(R.id.import_model);
        chooseImage = findViewById(R.id.choose_image);
        runModel = findViewById(R.id.run_model);

        createModeCards();
        importModel.setOnClickListener(view -> openModelPicker());
        chooseImage.setOnClickListener(view -> openImagePicker());
        runModel.setOnClickListener(view -> runInference());

        activeState = ModelImporter.activeState(modelsDirectory);
        VisionAdapter initialAdapter = activeState == null
                ? adapters.get(0)
                : AdapterRegistry.forId(activeState.adapterId());
        if (initialAdapter == null) {
            activeState = null;
            initialAdapter = adapters.get(0);
        }
        selectAdapter(initialAdapter, false);
    }

    private void createModeCards() {
        LayoutInflater inflater = LayoutInflater.from(this);
        for (VisionAdapter adapter : adapters) {
            View card = inflater.inflate(R.layout.view_mode_card, modeSelector, false);
            AdapterDefinition definition = adapter.definition();
            ((TextView) card.findViewById(R.id.mode_title)).setText(definition.titleResource());
            ((TextView) card.findViewById(R.id.mode_description)).setText(
                    definition.descriptionResource()
            );
            LinearLayout.LayoutParams layoutParams = adapters.size() == 2
                    ? new LinearLayout.LayoutParams(
                    0,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    1
            )
                    : new LinearLayout.LayoutParams(
                    dpToPixels(190),
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            layoutParams.setMarginEnd(dpToPixels(adapters.size() == 2 ? 6 : 12));
            card.setLayoutParams(layoutParams);
            card.setOnClickListener(view -> selectAdapter(adapter, true));
            modeSelector.addView(card);
            modeCards.put(definition.id(), card);
        }
    }

    private void selectAdapter(VisionAdapter adapter, boolean activateStoredModel) {
        if (selectedAdapter == adapter) {
            return;
        }
        closeRunner();
        selectedAdapter = adapter;
        activeState = ModelImporter.stateForAdapter(
                modelsDirectory,
                adapter.definition().id()
        );
        if (activateStoredModel && activeState != null) {
            try {
                ModelImporter.activate(modelsDirectory, activeState);
            } catch (Exception exception) {
                showError(getString(R.string.model_import_failed, exception.getMessage()));
                return;
            }
        }
        attachAdapterOptions();
        applyModeState();
        results.setText(activeState == null
                ? R.string.initial_instructions
                : adapter.definition().readyResultResource());
    }

    private void attachAdapterOptions() {
        adapterOptions.removeAllViews();
        selectedOptionsView = selectedAdapter.createOptionsView(
                LayoutInflater.from(this),
                adapterOptions
        );
        if (selectedOptionsView != null) {
            adapterOptions.addView(selectedOptionsView);
            adapterOptions.setVisibility(View.VISIBLE);
        } else {
            adapterOptions.setVisibility(View.GONE);
        }
    }

    private void openModelPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/octet-stream");
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
                closeRunner();
                ModelImporter.ImportResult imported = ModelImporter.importModel(
                        getApplicationContext(),
                        modelUri,
                        modelsDirectory
                );
                VisionAdapter importedAdapter = AdapterRegistry.forId(
                        imported.state().adapterId()
                );
                if (importedAdapter == null) {
                    throw new IllegalStateException("The imported model adapter is unavailable.");
                }
                runOnUiThread(() -> {
                    selectedAdapter = importedAdapter;
                    activeState = imported.state();
                    attachAdapterOptions();
                    applyModeState();
                    results.setText(importedAdapter.definition().readyResultResource());
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
        if (activeState == null
                || !activeState.adapterId().equals(selectedAdapter.definition().id())) {
            showError(getString(R.string.import_before_running));
            return;
        }
        if (selectedBitmap == null) {
            showError(getString(R.string.choose_before_running));
            return;
        }

        AdapterInput input;
        try {
            input = selectedAdapter.collectInput(selectedOptionsView);
        } catch (Exception exception) {
            showError(exception.getMessage());
            return;
        }
        setBusy(getString(selectedAdapter.definition().runningResource()));

        executor.execute(() -> {
            try {
                String output = runActiveModel(input);
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

    private String runActiveModel(AdapterInput input) throws Exception {
        if (activeRunner == null) {
            activeRunner = selectedAdapter.createRunner(
                    getApplicationContext(),
                    ModelImporter.modelFile(modelsDirectory, activeState.descriptor()),
                    activeState.descriptor()
            );
        }
        return activeRunner.run(selectedBitmap, input);
    }

    private void applyModeState() {
        for (Map.Entry<String, View> entry : modeCards.entrySet()) {
            entry.getValue().setBackgroundResource(
                    entry.getKey().equals(selectedAdapter.definition().id())
                            ? R.drawable.bg_mode_selected
                            : R.drawable.bg_mode_unselected
            );
        }

        AdapterDefinition definition = selectedAdapter.definition();
        importModel.setText(definition.importButtonResource());
        runModel.setText(definition.runButtonResource());
        if (activeState == null || !activeState.adapterId().equals(definition.id())) {
            modelStatus.setText(R.string.model_not_imported);
            modelExplanation.setText(definition.missingModelResource());
        } else {
            modelStatus.setText(activeState.descriptor().displayLabel());
            modelExplanation.setText(definition.readyModelResource());
        }
        updateControls(false);
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
        for (View card : modeCards.values()) {
            card.setEnabled(!busy);
        }
        importModel.setEnabled(!busy);
        chooseImage.setEnabled(!busy);
        setViewTreeEnabled(selectedOptionsView, !busy);
        runModel.setEnabled(!busy
                && activeState != null
                && activeState.adapterId().equals(selectedAdapter.definition().id())
                && selectedBitmap != null);
    }

    private static void setViewTreeEnabled(View view, boolean enabled) {
        if (view == null) {
            return;
        }
        view.setEnabled(enabled);
        if (view instanceof ViewGroup group) {
            for (int index = 0; index < group.getChildCount(); index++) {
                setViewTreeEnabled(group.getChildAt(index), enabled);
            }
        }
    }

    private void showError(String message) {
        setBusy(false);
        results.setText(message == null ? getString(R.string.inference_failed, "Unknown error") : message);
    }

    private void closeRunner() {
        if (activeRunner != null) {
            activeRunner.close();
            activeRunner = null;
        }
    }

    private int dpToPixels(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        closeRunner();
        if (selectedBitmap != null && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }
        super.onDestroy();
    }
}
