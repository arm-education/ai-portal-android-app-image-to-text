plugins {
    id("com.android.application")
}

android {
    namespace = "org.arm.learningpath.imageclassification"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.arm.learningpath.imageclassification"
        minSdk = 28
        targetSdk = 35
        testInstrumentationRunner = "android.test.InstrumentationTestRunner"
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        disable += "ChromeOsAbiSupport"
    }
}

// Framework test APIs are compile-only; the test APK requests their device libraries.
dependencies {
    androidTestCompileOnly(files(
        "${android.sdkDirectory}/platforms/android-35/optional/android.test.runner.jar",
        "${android.sdkDirectory}/platforms/android-35/optional/android.test.base.jar",
        "${android.sdkDirectory}/platforms/android-35/optional/android.test.mock.jar"
    ))
    implementation("com.google.ai.edge.litert:litert:2.1.6")
    implementation("org.pytorch:executorch-android:1.3.1")
}

apply(from = "generated-runtime-dependencies.gradle.kts")
