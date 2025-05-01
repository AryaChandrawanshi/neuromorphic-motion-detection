# Neuromorphic Motion Detection Using a Trained SNN in VHDL & Python

This project demonstrates a complete neuromorphic computing pipeline to recognize gestures using event-based data. We built a gesture classification model trained on the **DVSGesture dataset**, leveraging both **Spiking Neural Networks (SNN)** via `snnTorch` and traditional CNNs.

We simulate event-based (DVS-like) frames from a **standard webcam**, and perform **real-time prediction** using the trained model. The model is further optimized and **quantized using Brevitas**, with an attempt to convert it into a **hardware-compatible design using hls4ml**.

> ⚠️ **Note:** The hardware conversion step using `hls4ml` could not be completed due to library incompatibility and build errors during synthesis. This step remains a work in progress.

---

## Key Features

- 📡 **Gesture recognition using DVSGesture dataset**
- 🧠 **Neuromorphic model** built using `snnTorch` and CNNs
- ⚡ **Quantized model** using `Brevitas` for edge-device deployment
- 🛠️ **Attempted hardware synthesis** using `hls4ml`
- 🎥 **Real-time webcam input** converted to DVS-style event frames for live gesture detection

---

## Notebooks and Files

| File Name                         | Description                                                       |
|----------------------------------|-------------------------------------------------------------------|
| `nmc_model_cnn.ipynb`            | Main training notebook for CNN/SNN model using `snnTorch`         |
| `nmc_model_cnn_quantized.ipynb`  | Quantized model using `Brevitas`                                  |
| `nmc_model_cnn_hls.ipynb`        | Attempted hardware conversion via `hls4ml` *(incomplete due to errors)* |

---

## Dataset

- **DVSGesture** from `tonic.datasets`
- Event format: `(x, y, polarity, timestamp)`
- Converted into `(2, 128, 128)` tensors using ON/OFF polarity channels
- denoised, downsampled & transformed into frames of `(32, 2, 32, 32)` tensors

---

## Tools & Libraries Used

- `snnTorch` – Spiking neural network modeling
- `Brevitas` – Quantization-aware training
- `hls4ml` – *(Attempted)* HLS synthesis for FPGAs
- `Tonic` – Event data processing
- `OpenCV` – Webcam-based DVS frame simulation
- `PyTorch`, `NumPy`, `Matplotlib`

---

## Real-Time Gesture Detection

Standard webcam input is preprocessed into ON/OFF 2-channel event-like tensors by computing signed brightness differences. These are fed into the trained model to perform gesture classification in real time.

---

## Future Work

- Resolve `hls4ml` library issues for successful hardware conversion
- Add support for asynchronous event camera devices (e.g., DVS128 hardware)
- Optimize model for embedded or FPGA deployment

---
