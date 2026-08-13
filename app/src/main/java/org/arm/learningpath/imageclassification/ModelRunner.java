package org.arm.learningpath.imageclassification;

import android.graphics.Bitmap;

import java.io.Closeable;

interface ModelRunner extends Closeable {
    String run(Bitmap bitmap, AdapterInput input) throws Exception;

    @Override
    void close();
}
