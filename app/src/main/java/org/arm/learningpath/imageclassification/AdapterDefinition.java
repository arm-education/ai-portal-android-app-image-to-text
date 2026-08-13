package org.arm.learningpath.imageclassification;

record AdapterDefinition(
        String id,
        int titleResource,
        int descriptionResource,
        int importButtonResource,
        int runButtonResource,
        int missingModelResource,
        int readyModelResource,
        int readyResultResource,
        int runningResource
) {
}
