import torch
from models.ecg_cnn import ECGCNN
from models.compact_ecg_cnn import SmallECGCNN
from benchmark import get_benchmark_latency, get_model_size_mb
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

device = torch.device("cpu")

x = torch.randn(1, 1, 234).to(device)

# Baseline model
baseline_model = ECGCNN().to(device)
baseline_model.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/ecg_model.pth", map_location=device))
baseline_model.eval()

baseline_latency = get_benchmark_latency(baseline_model, x)
torch.save(baseline_model.state_dict(), "Results/baseline_temp.pth")
baseline_size = get_model_size_mb("Results/baseline_temp.pth")

print(f"Baseline Size: {baseline_size:.4f} MB")
print(f"Baseline Latency: {baseline_latency:.4f} ms")

# Baseline model with patient-wise split
patient_wise_model = ECGCNN().to(device)
patient_wise_model.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/ecg_model_patient_split.pth", map_location=device))
patient_wise_model.eval()

patient_wise_split_latency = get_benchmark_latency(patient_wise_model, x)
torch.save(patient_wise_model.state_dict(), "Results/patient_wise_split_temp.pth")
patient_wise_split_size = get_model_size_mb("Results/patient_wise_split_temp.pth")

print(f"Patient Wise Split Size: {patient_wise_split_size:.4f} MB")
print(f"Patient Wise Split Latency: {patient_wise_split_latency:.4f} ms")

# Smaller ECG model with patient-wise split
smaller_model = SmallECGCNN().to(device)
smaller_model.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/compact_patient_split.pth", map_location=device))
smaller_model.eval()

smaller_ecg_latency = get_benchmark_latency(smaller_model, x)
torch.save(smaller_model.state_dict(), "Results/compact_patient_split_temp.pth")
smaller_ecg_size = get_model_size_mb("Results/compact_patient_split_temp.pth")

print(f"Small Model Size: {smaller_ecg_size:.4f} MB")
print(f"Small Model Latency: {smaller_ecg_latency:.4f} ms")

# Quantized model
quant_model = ECGCNN().to(device)
torch.backends.quantized.engine = 'qnnpack'
quant_model.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/ecg_model_patient_split.pth", map_location=device))
quant_model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    quant_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "Results/quantized_ecg_model.pth")

quant_latency = get_benchmark_latency(quantized_model, x)
torch.save(quantized_model.state_dict(), "Results/quant_temp.pth")
quant_size = get_model_size_mb("Results/quant_temp.pth")

print(f"Quantized size: {quant_size:.4f} MB")
print(f"Quantized latency: {quant_latency:.4f} ms")

# Quantized smaller ecg model
smaller_model = SmallECGCNN().to(device)
torch.backends.quantized.engine = 'qnnpack'
smaller_model.load_state_dict(torch.load("/Users/Chakradhar/PycharmProjects/ECG_Project/Results/compact_patient_split.pth", map_location=device))
smaller_model.eval()

quant_smaller_model = torch.quantization.quantize_dynamic(
    smaller_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(quant_smaller_model.state_dict(), "Results/quantized_smaller_ecg_model.pth")

quant_smaller_latency = get_benchmark_latency(quant_smaller_model, x)
torch.save(quant_smaller_model.state_dict(), "Results/quant_smaller_temp.pth")
quant_smaller_size = get_model_size_mb("Results/quant_smaller_temp.pth")

print(f"Quantized Small Model size: {quant_smaller_size:.4f} MB")
print(f"Quantized Small Model latency: {quant_smaller_latency:.4f} ms")
