import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ecg_cnn import ECGCNN
from models.compact_ecg_cnn import SmallECGCNN
from utils.data_utils import load_data, to_tensor
from utils.train_utils import evaluate
from utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_noise(X, noise_level=0.1):
    noise_value = noise_level * torch.randn_like(X)
    return X + noise_value


def evaluate_model(model, loader):
    preds, labels = evaluate(model, loader, device)
    acc, f1 = compute_metrics(labels, preds)
    return acc, f1


def run_robustness_test(model_path, model_type, loader, X_data, y_data, noise_level=0.1):
    if model_type == "full":
        model = ECGCNN().to(device)
    else:
        model = SmallECGCNN().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n{model_type.upper()} MODEL (noise={noise_level})")

    # Clean performance
    clean_acc, clean_f1 = evaluate_model(model, loader)
    print("CLEAN DATA PERFORMANCE")
    print(f"Accuracy: {clean_acc:.2f}")
    print(f"F1 Score: {clean_f1:.2f}")

    # Noisy performance
    X_test_noisy = add_noise(X_data, noise_level)
    noisy_dataset = TensorDataset(X_test_noisy, y_data)
    noisy_loader = DataLoader(noisy_dataset, batch_size=64, shuffle=False)

    noisy_acc, noisy_f1 = evaluate_model(model, noisy_loader)
    print("NOISY DATA PERFORMANCE")
    print(f"Accuracy: {noisy_acc:.2f}")
    print(f"F1 Score: {noisy_f1:.2f}")

    # Drop
    print("ROBUSTNESS DROP")
    print(f"Accuracy Drop: {clean_acc - noisy_acc:.2f}")
    print(f"F1 Drop: {clean_f1 - noisy_f1:.2f}")


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    X_test, y_test = to_tensor(X_test, y_test)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    torch.manual_seed(42)

    noise_levels = [0.05, 0.1, 0.2, 0.3]

    for noise in noise_levels:
        run_robustness_test(
            "Results/ecg_model_patient_split.pth",
            model_type="full",
            loader=test_loader,
            X_data=X_test,
            y_data=y_test,
            noise_level=noise
        )

    for noise in noise_levels:
        run_robustness_test(
            "Results/compact_patient_split.pth",
            model_type="small",
            loader=test_loader,
            X_data=X_test,
            y_data=y_test,
            noise_level=noise
        )
