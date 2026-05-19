import os
import torch
import time


def get_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def get_benchmark_latency(ecg_model, input_tensor, runs=200):
    ecg_model.eval()
    with torch.no_grad():
        for _ in range(20):
            _ = ecg_model(input_tensor)

        start = time.perf_counter()

        for _ in range(runs):
            _ = ecg_model(input_tensor)

        end = time.perf_counter()

    return (end - start) / runs * 1000
