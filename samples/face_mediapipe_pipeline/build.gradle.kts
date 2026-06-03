plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val faceMediaPipePipelineUrl = "https://huggingface.co/picoxr/face-mediapipe-pipeline/resolve/main/face-mediapipe-pipeline.zip?download=true"
val faceMediaPipePipelineAssetDir = layout.projectDirectory.dir("src/main/assets/face-mediapipe-pipeline")
val faceMediaPipePipelineZip = layout.buildDirectory.file("downloads/face-mediapipe-pipeline.zip")

val downloadFaceMediaPipePipelineAssets by tasks.registering {
    inputs.property("sourceUrl", faceMediaPipePipelineUrl)
    outputs.dir(faceMediaPipePipelineAssetDir)

    doLast {
        val zipFile = faceMediaPipePipelineZip.get().asFile
        zipFile.parentFile.mkdirs()

        ant.invokeMethod("get", mapOf("src" to faceMediaPipePipelineUrl, "dest" to zipFile))
        delete(faceMediaPipePipelineAssetDir)
        copy {
            from(zipTree(zipFile)) {
                eachFile {
                    path = path.removePrefix("face-mediapipe-pipeline/")
                }
                includeEmptyDirs = false
            }
            into(faceMediaPipePipelineAssetDir)
        }
    }
}

android {
    compileSdk = 34
    ndkVersion = "26.3.11579264"
    namespace = "com.bytedance.pico.secure_mr_demo.face_mediapipe_pipeline"

    defaultConfig {
        minSdk = 34
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        applicationId = "com.bytedance.pico.secure_mr_demo.face_mediapipe_pipeline"

        externalNativeBuild {
            cmake {
                arguments.add("-DANDROID_STL=c++_shared")
                arguments.add("-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=OFF")
            }
            ndk {
                abiFilters.add("arm64-v8a")
            }
        }
    }

    lint {
        disable.add("ExpiredTargetSdkVersion")
    }

    buildFeatures {
        prefab = true
    }

    buildTypes {
        debug {
            isDebuggable = true
            isJniDebuggable = true
        }
        release {
            isDebuggable = false
            isJniDebuggable = false
        }
    }

    externalNativeBuild {
        cmake {
            version = "3.22.1"
            path("CMakeLists.txt")
        }
    }

    sourceSets {
        getByName("main") {
            manifest.srcFile("AndroidManifest.xml")
        }
    }

    packaging {
        jniLibs {
            keepDebugSymbols.add("**.so")
        }
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }.configureEach {
    dependsOn(downloadFaceMediaPipePipelineAssets)
}
