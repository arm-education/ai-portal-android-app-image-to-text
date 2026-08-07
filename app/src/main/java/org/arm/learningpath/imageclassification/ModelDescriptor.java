package org.arm.learningpath.imageclassification;

record ModelDescriptor(
        String id,
        String displayName,
        ModelTask task,
        ModelRuntime runtime,
        String fileName,
        LiteRtClassifier.PreprocessingProfile preprocessingProfile
) {
    enum ModelRuntime {
        LITERT("LiteRT"),
        EXECUTORCH("ExecuTorch");

        private final String displayName;

        ModelRuntime(String displayName) {
            this.displayName = displayName;
        }

        String displayName() {
            return displayName;
        }
    }

    String displayLabel() {
        return displayName + " · " + runtime.displayName();
    }
}
