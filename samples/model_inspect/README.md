# Sample: Model Inspect Tool

This sample runs a single SecureMR model inference to validate a serialized model on device. The app does not render any content; it loads the model and inputs from paths provided through system properties and dumps the outputs to the app's internal storage for inspection.

## Usage

1. Push your model files to the device and set the properties before launching the app:

```
adb push model.serialized.bin /data/local/tmp/my_model/
adb push mode.serialized.json /data/local/tmp/my_model/   # falls back to model.serialized.json if needed
adb shell setprop debug.securemr.model_inspect.model_dir /data/local/tmp/my_model

# Optional input file applied to all model inputs (random data is generated if omitted)
adb push input.bin /data/local/tmp/my_model/input.bin
adb shell setprop debug.securemr.model_inspect.input /data/local/tmp/my_model/input.bin
```

2. Build and install:

```
./gradlew :samples:model_inspect:installDebug
```

3. Launch the app. Logs will describe the parsed tensors, input loading/generation, pipeline submission, and readback.

## Outputs

- Outputs are read back via the SecureMR readback API and written under the app's external files directory (for example `/sdcard/Android/data/com.bytedance.pico.secure_mr_demo.model_inspect/files/model_inspect/`) as `model_inspect_output_<tensor>.bin`.
- Logcat will print a short preview of each output along with the tensor shape and data type for quick verification.
