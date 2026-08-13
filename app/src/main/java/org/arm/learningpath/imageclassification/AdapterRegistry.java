package org.arm.learningpath.imageclassification;

import java.util.ArrayList;
import java.util.List;

final class AdapterRegistry {
    private static final List<VisionAdapter> ADAPTERS = createAdapters();

    private AdapterRegistry() {
    }

    static List<VisionAdapter> all() {
        return ADAPTERS;
    }

    static VisionAdapter forId(String id) {
        for (VisionAdapter adapter : ADAPTERS) {
            if (adapter.definition().id().equals(id)) {
                return adapter;
            }
        }
        return null;
    }

    private static List<VisionAdapter> createAdapters() {
        List<VisionAdapter> adapters = new ArrayList<>();
        adapters.add(new LiteRtImageClassificationAdapter());
        adapters.add(new ExecuTorchClipAdapter());
        adapters.addAll(GeneratedAdapterRegistry.adapters());
        return List.copyOf(adapters);
    }
}
