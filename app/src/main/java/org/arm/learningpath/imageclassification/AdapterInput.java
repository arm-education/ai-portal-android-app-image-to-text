package org.arm.learningpath.imageclassification;

import java.util.List;

interface AdapterInput {
}

record EmptyAdapterInput() implements AdapterInput {
}

record DescriptionAdapterInput(List<String> descriptions) implements AdapterInput {
    DescriptionAdapterInput {
        descriptions = List.copyOf(descriptions);
    }
}
