# ESP-Fi-HAR Fall Detection — Training and Deployment Settings

## 1. Project objective

Train a binary Wi-Fi CSI classifier:

- `0` = non-fall activity
- `1` = fall

The model is trained once using the ESP-Fi-HAR dataset and then exported to ONNX for use by the dashboard backend.

## 2. Dataset used

Dataset: [ESP-Fi-HAR](https://github.com/AutoSmartGroup/ESP-Fi-HAR)

The dataset contains CSI amplitude measurements collected using ESP32-based hardware.

### Dataset input format

| Setting | Value |
|---|---|
| Modality | CSI amplitude only |
| Actual model input | `950 × 52` |
| Time dimension | 950 frames |
| Subcarrier dimension | 52 active subcarriers |
| Data file format | MATLAB `.mat` |
| MATLAB variable | `CSIamp` |
| Supported raw shapes | `(950, 52)` or `(52, 950)` |
| Normalization | Per-sample Z-score |

The notebook converts `(52, 950)` to `(950, 52)` when required.

> Note: The dataset repository documents the CSI input as `1 × 950 × 52`. The training notebook uses the equivalent squeezed form `(950, 52)` before the model transposes it for convolution.

## 3. Activities and labels

The original dataset has seven activities:

- `arm_wave`
- `fall`
- `jump`
- `run`
- `squat`
- `turn`
- `walk`

For this project they are mapped to two classes:

| Original activity | Model label |
|---|---:|
| `fall` | `1` — fall |
| `arm_wave`, `jump`, `run`, `squat`, `turn`, `walk` | `0` — non-fall |

## 4. Dataset directory layout

The expected Kaggle layout is:

```text
ESP-Fi-HAR/
└── Model Code/
    └── Data/
        └── ESP-Fi_HAR/
            ├── train_amp/
            │   ├── arm_wave/
            │   ├── fall/
            │   ├── jump/
            │   ├── run/
            │   ├── squat/
            │   ├── turn/
            │   └── walk/
            └── test_amp/
                ├── arm_wave/
                ├── fall/
                ├── jump/
                ├── run/
                ├── squat/
                ├── turn/
                └── walk/
```

The notebook downloads `ESP-Fi_dataset.rar`, extracts it, and searches for the `train_amp` directory.

## 5. Preprocessing

For every sample:

```python
x = (x - x.mean()) / (x.std() + 1e-8)
```

This is per-sample normalization. The same operation must be performed on live ESP32 data before inference.

No phase data is used by this training pipeline.

## 6. Model architecture

Model name: `ESP32FallNet`

### Input and output

```text
Input:  (batch, 950, 52)
Output: (batch, 2)
```

Output order:

```text
output[0] = non_fall_logit
output[1] = fall_logit
```

The backend applies softmax and reads class index `1` as the fall probability.

### Network parameters

| Parameter | Value |
|---|---:|
| Initial projection | `Conv1d(52, 48, kernel_size=1)` |
| TCN channel width | 48 |
| TCN dilation values | `1, 2, 4, 8, 16, 32, 64, 128, 256` |
| TCN kernel size | 3 |
| Convolution type | Depthwise convolution + pointwise convolution |
| Activation | GELU |
| Normalization | BatchNorm1d |
| Temporal pooling | Maximum pooling and last-frame feature |
| Hidden fully connected size | 32 |
| Final layer | 2 logits |

The model does not contain a final sigmoid. `CrossEntropyLoss` is used during training, and softmax is applied during inference.

## 7. Training settings

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Device | CUDA when available, otherwise CPU |
| Batch size | 64 in the notebook currently used |
| Epochs | 15 |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Loss | CrossEntropyLoss |
| Number of classes | 2 |
| Shuffle training data | Yes |
| Test shuffle | No |
| Random seed | Not fixed in the original training cell; set one before final experiment for reproducibility |

The test set should be used only for final reporting. It should not be used to choose the best epoch.

## 8. ONNX export

The notebook exports:

```text
esp32_fallnet.onnx
esp32_fallnet.onnx.data
```

Both files must remain in the same directory. The `.onnx.data` file contains external model weights and is required.

### ONNX contract

```text
Input name:  csi_input
Input shape: [batch, 950, 52]
Input type:  float32

Output name: fall_logits
Output shape: [batch, 2]
Output type: float32
```

## 9. Required live ESP32 format

The trained model expects the live stream to match the dataset:

```text
One frame:   52 amplitude values
One window:  950 frames
Model input: 950 × 52 float32 values
```

The live pipeline must therefore:

1. Read 52 amplitude values from every CSI frame.
2. Buffer 950 consecutive frames.
3. Apply per-window or per-sample Z-score normalization consistently with training.
4. Send the tensor to ONNX Runtime.
5. Apply softmax to the two output logits.
6. Use output index `1` as the fall probability.

## 10. Important difference from the current firmware

The current firmware/dashboard configuration was built around:

```text
500 frames × 232 values
116 amplitude values + 116 phase values
```

That format is **not compatible** with this new ESP-Fi-HAR model.

Before using `esp32_fallnet.onnx`, update both sides:

- ESP32 firmware: send 52 amplitude values per frame.
- Dashboard backend: use `WINDOW_SIZE = 950` and `FEATURE_COUNT = 52`.
- Backend preprocessing: use the same Z-score normalization as the notebook.
- Backend model path: point to `esp32_fallnet.onnx`.
- Keep `esp32_fallnet.onnx.data` beside the ONNX file.

## 11. Model placement

Place both exported files in the backend model directory, for example:

```text
dashboard/backend/
├── esp32_fallnet.onnx
└── esp32_fallnet.onnx.data
```

Update `app.py` so its model path points to `esp32_fallnet.onnx`.

## 12. Verification checklist

Before testing a fall:

- [ ] The dataset loader reports a non-zero number of train and test samples.
- [ ] Each loaded sample has shape `(950, 52)`.
- [ ] The ONNX model loads without a shape-inference error.
- [ ] ONNX input is `[batch, 950, 52]`.
- [ ] ONNX output is `[batch, 2]`.
- [ ] Both `.onnx` and `.onnx.data` are in the same folder.
- [ ] Firmware sends 52 amplitude values per frame.
- [ ] Backend buffers 950 frames.
- [ ] Backend uses the same Z-score preprocessing.
- [ ] A fall probability is calculated from output index `1`.

