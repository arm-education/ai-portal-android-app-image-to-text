package org.arm.learningpath.imageclassification;

import android.content.ContentProvider;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.pm.ProviderInfo;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.test.InstrumentationTestCase;
import android.test.mock.MockContentResolver;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Set;

/** Device tests use public model/image fixtures staged in files/model-test-inputs. */
public final class ModelSupportTest extends InstrumentationTestCase {
    public void testRegistryHasUniqueNames() {
        Set<String> ids = new HashSet<>();
        Set<String> names = new HashSet<>();
        for (VisionAdapter adapter : AdapterRegistry.all()) {
            for (ModelDescriptor model : ModelRegistry.forAdapter(adapter.definition().id())) {
                assertTrue(model.id(), ids.add(model.id()));
                assertTrue(model.fileName(), names.add(model.fileName().toLowerCase(java.util.Locale.ROOT)));
            }
        }
    }

    public void testImportAndClassifyModels() throws Exception {
        Context target = getInstrumentation().getTargetContext();
        File fixtures = new File(target.getFilesDir(), "model-test-inputs");
        assertTrue("Stage model-test-inputs as described beside this test", fixtures.isDirectory());
        Bitmap dog = BitmapFactory.decodeFile(new File(fixtures, "dog.jpg").getPath());
        Bitmap cat = BitmapFactory.decodeFile(new File(fixtures, "cat.jpg").getPath());
        assertNotNull(dog);
        assertNotNull(cat);
        MockContentResolver resolver = new MockContentResolver(target);
        FixtureProvider provider = new FixtureProvider(fixtures);
        ProviderInfo providerInfo = new ProviderInfo();
        providerInfo.authority = "model-fixtures";
        providerInfo.exported = true;
        provider.attachInfo(target, providerInfo);
        resolver.addProvider("model-fixtures", provider);
        Context context = new ContextWrapper(target) {
            @Override public ContentResolver getContentResolver() { return resolver; }
        };
        File installed = new File(target.getCacheDir(), "model-support-test");
        int checked = 0;
        int vivoChecked = 0;
        try {
            for (VisionAdapter adapter : AdapterRegistry.all()) {
                for (ModelDescriptor model : ModelRegistry.forAdapter(adapter.definition().id())) {
                    File source = new File(fixtures, model.fileName());
                    if (model.id().endsWith("-vivo-executorch")) {
                        assertTrue("Missing required fixture " + source, source.isFile());
                    }
                    if (!source.isFile()) continue;
                    Log.i("ModelSupportTest", "START " + model.id());
                    ModelImporter.ImportResult imported = ModelImporter.importModel(context,
                            Uri.parse("content://model-fixtures/" + model.fileName()), installed);
                    assertEquals(model.id(), imported.state().descriptor().id());
                    assertEquals(model.id(), ModelImporter.activeState(installed).descriptor().id());
                    try (ModelRunner runner = adapter.createRunner(target,
                            ModelImporter.modelFile(installed, model), model)) {
                        boolean clip = ExecuTorchClipAdapter.ID.equals(model.adapterId());
                        AdapterInput input = clip
                                ? new DescriptionAdapterInput(java.util.List.of("a dog", "a cat", "a bicycle"))
                                : adapter.collectInput(null);
                        String first = runner.run(dog, input);
                        String second = runner.run(cat, input);
                        Log.i("ModelSupportTest", model.id() + " DOG " + first);
                        Log.i("ModelSupportTest", model.id() + " CAT " + second);
                        if (clip) {
                            assertTrue(first, first.split("\n")[2].contains("a dog"));
                            assertTrue(second, second.split("\n")[2].contains("a cat"));
                        } else {
                            assertTrue(model.id() + first, first.contains("Samoyed"));
                            assertTrue(model.id() + second, second.matches("(?s).*(Persian cat|Siamese cat|tabby|tiger cat|Egyptian cat).*"));
                        }
                        assertFalse(model.id(), first.equals(second));
                    }
                    checked++;
                    if (model.id().endsWith("-vivo-executorch")) vivoChecked++;
                    deleteTree(installed);
                    Log.i("ModelSupportTest", "PASS " + model.id());
                }
            }
            assertEquals("All eight vivo models must run", 8, vivoChecked);
            Log.i("ModelSupportTest", "TOTAL " + checked);
        } finally {
            dog.recycle(); cat.recycle(); deleteTree(installed);
        }
    }

    private static void deleteTree(File file) {
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        if (file.exists() && !file.delete()) throw new IllegalStateException("Cannot remove " + file);
    }

    private static final class FixtureProvider extends ContentProvider {
        private final File directory;
        FixtureProvider(File directory) { this.directory = directory; }
        @Override public boolean onCreate() { return true; }
        @Override public Cursor query(Uri uri, String[] projection, String selection,
                                      String[] args, String order) {
            MatrixCursor cursor = new MatrixCursor(new String[]{OpenableColumns.DISPLAY_NAME});
            cursor.addRow(new Object[]{uri.getLastPathSegment()});
            return cursor;
        }
        @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
            return ParcelFileDescriptor.open(new File(directory, uri.getLastPathSegment()),
                    ParcelFileDescriptor.MODE_READ_ONLY);
        }
        @Override public String getType(Uri uri) { return "application/octet-stream"; }
        @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
        @Override public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException(); }
        @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException(); }
    }
}
