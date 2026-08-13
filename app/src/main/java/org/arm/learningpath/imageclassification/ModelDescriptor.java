package org.arm.learningpath.imageclassification;

record ModelDescriptor(
        String id,
        String displayName,
        String adapterId,
        String runtimeDisplayName,
        String fileName
) {
    String displayLabel() {
        return displayName + " · " + runtimeDisplayName;
    }
}
