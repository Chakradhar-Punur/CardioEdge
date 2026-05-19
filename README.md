# CardioEdge

Efficient and Robust ECG Arrhythmia Classification for Wearable Edge Deployment

## Overview

CardioEdge is an end-to-end deep learning pipeline for ECG arrhythmia classification designed for real-world wearable healthcare deployment. The project uses 1D Convolutional Neural Networks (CNNs) trained on the MIT-BIH Arrhythmia Dataset and focuses on realistic patient-wise evaluation, robustness under wearable ECG noise conditions, and lightweight edge deployment optimization.

The system also includes a real-time Streamlit dashboard supporting ECG visualization, live inference, confidence estimation, and Apple Watch ECG integration workflows.

---

## Features

- ECG heartbeat segmentation and preprocessing
- Patient-wise train/validation/test splitting
- 1D CNN and compact CNN architectures
- Dynamic INT8 quantization for model compression
- Robustness testing under noisy ECG conditions
- Real-time ECG AI dashboard using Streamlit
- Apple Watch ECG workflow integration
- Deployment-oriented latency and model size benchmarking

---

## Dataset

This project uses the MIT-BIH Arrhythmia Dataset:

- 48 ECG recordings
- 109,209 extracted heartbeat segments
- 5 AAMI arrhythmia classes
- 360 Hz sampling frequency

Dataset Link:
https://physionet.org/content/mitdb/1.0.0/

---

## Model Performance

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN (Random Split) | ~99% | Artificially Inflated |
| Patient-wise CNN | 73% | 0.44 |
| Small CNN | 67% | 0.47 |
| Quantized Small CNN | ~67% | ~0.46 |

---

## Optimization Results

- Model compression: 0.94 MB → 0.07 MB
- ~92% reduction in model size
- Sub-millisecond inference latency
- Improved robustness under wearable ECG noise conditions

---

## Dashboard

The Streamlit dashboard supports:

- ECG waveform visualization
- Live arrhythmia prediction
- Confidence distribution analysis
- Robustness testing with noise injection
- Apple Watch ECG workflow demonstration

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Chakradhar-Punur/CardioEdge.git
cd CardioEdge
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run ECG preprocessing before training:

```bash
python preprocessing.py
```

Train the baseline ECG CNN model:

```bash
python train.py
```

(Optional) Train the compact Small CNN model:

```bash
python train_smaller_ecg_cnn.py
```

(Optional) Run robustness testing:

```bash
python robustness.py
```

(Optional) Run dynamic INT8 quantization and optimization:

```bash
python optimize.py
```

Launch the Streamlit ECG AI dashboard:

```bash
streamlit run app.py
```

The dashboard will open locally in your browser at:

```text
http://localhost:8501
```

---

## Project Structure

```text
CardioEdge/
│
├── app.py                          # Streamlit ECG AI dashboard
├── train.py                        # Baseline and patient-wise CNN training
├── train_smaller_ecg_cnn.py        # Compact Small CNN training
├── preprocessing.py                # ECG preprocessing and heartbeat extraction
├── optimize.py                     # Dynamic INT8 quantization and optimization
├── robustness.py                   # Noise robustness testing
├── benchmark.py                    # Latency and model benchmarking
├── main.py                         # Main execution script
├── watch_pdf_to_csv.py             # Apple Watch ECG PDF to CSV conversion
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/                         # CNN model definitions
├── utils/                          # Helper and utility functions
├── data/                           # Dataset storage
├── Results/                        # Saved models and evaluation outputs
│
├── watch_ecg.pdf                   # Apple Watch ECG export
├── watch_ecg.csv                   # Converted ECG CSV signal
├── watch_ecg.png                   # ECG waveform visualization
```

---

## Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- Matplotlib
- WFDB
- Scikit-learn

---

## Future Work

- Real-time Apple Watch ECG streaming
- Swift/Xcode native deployment
- Quantization-aware training
- Transformer-based ECG models
- Improved robustness and false positive reduction

---

## Author

Chakradhar Punur  
Rutgers University – MS in Computer Science