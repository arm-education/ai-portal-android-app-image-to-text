package org.arm.learningpath.imageclassification;

import android.graphics.Bitmap;

import java.io.Closeable;
import java.util.List;

interface ModelRunner extends Closeable {
    String run(Bitmap bitmap, List<String> descriptions) throws Exception;

    @Override
    void close();
}
