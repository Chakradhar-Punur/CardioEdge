import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from models.ecg_cnn import ECGCNN
from models.compact_ecg_cnn import SmallECGCNN
from utils.data_utils import load_data, to_tensor

st.set_page_config(page_title="CardioEdge", layout="wide")
device = torch.device("cpu")

CLASS_MAP = {
    0: "F (Fusion)",
    1: "N (Normal)",
    2: "Q (Unknown)",
    3: "S (Supraventricular)",
    4: "V (Ventricular)"
}

st.markdown("""
<style>
body { background-color: #0b0b0b; color: white; }
.card {
    background-color: #1c1c1e;
    padding: 20px;
    border-radius: 15px;
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    full = ECGCNN().to(device)
    full.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/ecg_model_patient_split.pth", map_location=device))
    full.eval()

    small = SmallECGCNN().to(device)
    small.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/compact_patient_split.pth", map_location=device))
    small.eval()

    return full, small


full_model, small_model = load_models()
st.title("CardioEdge - Real-time Cardiac Analysis")
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Inference",
    "Preprocessing",
    "Training Results",
    "Optimization"
])


with tab1:
    st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox("Model", ["Small CNN (Recommended)", "Full CNN"])
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.0)

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Demo ECGs")

demo_choice = st.sidebar.radio(
    "Load Demo ECG",
    ["None", "Normal ECG", "Abnormal ECG", "Noisy ECG"]
)

X_train, X_val, X_test, y_train, y_val, y_test = load_data()
X_test, y_test = to_tensor(X_test, y_test)


def find_sample_index(target_type="normal"):
    matching_indices = []

    for i in range(len(y_test)):
        label_value = int(y_test[i])

        if target_type == "normal" and label_value == 1:
            matching_indices.append(i)

        if target_type == "abnormal" and label_value != 1:
            matching_indices.append(i)

    if matching_indices:
        return random.choice(matching_indices)

    return 0


signal = None
label = None

if uploaded_file is not None:
    data = np.loadtxt(uploaded_file, delimiter=",")
    data = data[:234]
    signal = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    label = "Uploaded ECG"

elif demo_choice != "None":

    if demo_choice == "Normal ECG":
        idx = find_sample_index("normal")

    elif demo_choice == "Abnormal ECG":
        idx = find_sample_index("abnormal")

    else:
        idx = find_sample_index("normal")

    signal = X_test[idx].unsqueeze(0)
    label = CLASS_MAP[y_test[idx].item()]

    if demo_choice == "Noisy ECG":
        signal = signal + 0.3 * torch.randn_like(signal)
        label = "Noisy ECG"

with tab1:
    if signal is None:
        st.markdown("""
        <div style='text-align:center; padding-top:120px;'>
            <h1 style='font-size:60px;'>CardioEdge</h1>
            <h3 style='color:gray;'>Upload ECG or choose demo sample</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        signal = signal.to(device)
        if noise_level > 0 and demo_choice != "Noisy ECG":
            signal = signal + noise_level * torch.randn_like(signal)

        model = small_model if "Small" in model_choice else full_model

        with torch.no_grad():
            output = model(signal)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            confidence = probs[pred]

        col1, col2 = st.columns([1.6, 1])

        with col1:
            st.subheader("ECG_Signal")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(signal.cpu().numpy().flatten())
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.set_title("ECG Waveform")
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.subheader("Prediction")
            class_name = CLASS_MAP.get(int(pred), "Unknown")
            st.metric("Class", class_name)
            st.metric("Confidence", f"{confidence:.2f}")
            st.markdown("---")
            st.write("True Label:", label)
            st.markdown("---")

            if label not in ["Uploaded ECG", "Noisy ECG"]:
                if class_name == label:
                    st.success("Correct Prediction")
                else:
                    st.error("Incorrect Prediction")
            elif label == "Noisy ECG":
                st.warning("Robustness Test Mode")

            st.subheader("Confidence Distribution")
            fig2, ax2 = plt.subplots(figsize=(4, 2.5))
            ax2.bar(list(CLASS_MAP.values()), probs.tolist())
            plt.xticks(rotation=45)
            st.pyplot(fig2)


with tab2:
    st.header("ECG Preprocessing Pipeline")

    st.subheader("Dataset Information")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.metric("Sampling Frequency", "360 Hz")
        st.metric("Signal Shape", "(650000, 2)")
        st.metric("Extracted Beats", "109,209")

    with col_info2:
        st.metric("Signal Channels", "MLII, V5")
        st.metric("Beat Length", "234")
        st.metric("Final Dataset Shape", "(109209, 234)")

    st.subheader("Class Distribution")

    class_distribution = {
        "Class": [
            "N (Normal)",
            "Q (Unknown)",
            "V (Ventricular)",
            "S (Supraventricular)",
            "F (Fusion)"
        ],
        "Samples": [90350, 8041, 7235, 2781, 802]
    }

    st.dataframe(class_distribution)

    fig_dist, ax_dist = plt.subplots()
    ax_dist.bar(
        class_distribution["Class"],
        class_distribution["Samples"]
    )
    ax_dist.set_ylabel("Number of Samples")
    ax_dist.set_title("ECG Class Distribution")
    plt.xticks(rotation=15)
    st.pyplot(fig_dist)

    st.info(
        "The MIT-BIH Arrhythmia dataset is highly imbalanced, with normal ECG beats dominating the dataset."
    )

    st.markdown("---")

    st.subheader("Dataset Split Information")

    col_split1, col_split2, col_split3 = st.columns(3)

    with col_split1:
        st.metric("Train Patients", "33")
        st.metric("Train Samples", "75,264")

    with col_split2:
        st.metric("Validation Patients", "7")
        st.metric("Validation Samples", "16,683")

    with col_split3:
        st.metric("Test Patients", "8")
        st.metric("Test Samples", "17,262")

    st.markdown("---")

    st.subheader("Raw vs Processed ECG")

    raw_signal = X_test[0].cpu().numpy().flatten()

    processed_signal = (
        (raw_signal - np.mean(raw_signal)) /
        (np.std(raw_signal) + 1e-8)
    )

    processed_signal = processed_signal * 0.6

    fig3, ax3 = plt.subplots()
    ax3.plot(raw_signal, label="Raw ECG", linewidth=3, alpha=0.9)
    ax3.plot(processed_signal, label="Processed ECG", linewidth=2, linestyle="--")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("ECG Preprocessing")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    st.markdown("""
    ### Preprocessing Steps
    - ECG signal normalization
    - Beat segmentation
    - Window extraction (234 samples)
    - Patient-wise train/validation/test split
    - Tensor conversion for CNN input
    """)

with tab3:
    st.header("Training Results")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Patient-wise Accuracy", "73%")
        st.metric("Patient-wise F1", "0.44")

    with col_b:
        st.metric("Small CNN Accuracy", "67%")
        st.metric("Small CNN F1", "0.47")

    with col_c:
        st.metric("Baseline Accuracy", "99%")
        st.metric("Baseline Split", "Random")

    st.subheader("Model Comparison")

    comparison_data = {
        "Model": [
            "Baseline CNN",
            "Patient-wise CNN",
            "Small CNN",
            "Quantized Small CNN"
        ],
        "Accuracy": [0.99, 0.73, 0.67, 0.67],
        "Model Size (MB)": [0.94, 0.94, 0.24, 0.07],
        "Latency (ms)": [0.15, 0.15, 0.14, 0.21]
    }

    st.dataframe(comparison_data)

with tab4:
    st.header("Model Optimization")

    st.markdown("""
    ### Optimization Techniques Applied

    - Dynamic Quantization
    - Compact CNN Architecture
    - Patient-wise Evaluation
    - Robustness Testing with Noise
    - Latency and Size Benchmarking
    """)

    models = [
        "Baseline",
        "Patient-wise",
        "Small CNN",
        "Quantized Small"
    ]

    sizes = [0.94, 0.94, 0.24, 0.07]

    fig4, ax4 = plt.subplots()
    ax4.bar(models, sizes)
    ax4.set_ylabel("Model Size (MB)")
    ax4.set_title("Model Compression Comparison")
    plt.xticks(rotation=15)
    st.pyplot(fig4)

    st.success("Quantized Small CNN achieved ~92% model size reduction compared to baseline.")
