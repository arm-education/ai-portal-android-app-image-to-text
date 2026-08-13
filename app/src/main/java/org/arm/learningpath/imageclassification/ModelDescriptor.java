package org.arm.learningpath.imageclassification;

record ModelDescriptor(
        String id,
        String displayName,
        String adapterId,
        String runtimeDisplayName,
        String fileName,
        String configurationId
) {
    ModelDescriptor(
            String id,
            String displayName,
            String adapterId,
            String runtimeDisplayName,
            String fileName
    ) {
        this(id, displayName, adapterId, runtimeDisplayName, fileName, "");
    }

    String displayLabel() {
        return displayName + " · " + runtimeDisplayName;
    }
}
