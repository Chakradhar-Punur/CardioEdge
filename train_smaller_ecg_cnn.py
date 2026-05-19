import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.compact_ecg_cnn import SmallECGCNN
from utils.data_utils import load_data, to_tensor
from utils.train_utils import train_one_epoch, validate, evaluate
from utils.metrics import compute_metrics, get_confusion_matrix, plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

X_train, y_train = to_tensor(X_train, y_train)
X_val, y_val = to_tensor(X_val, y_val)
X_test, y_test = to_tensor(X_test, y_test)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

class_counts = np.bincount(y_train.cpu().numpy())
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SmallECGCNN().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 5

print("Training Small ECG CNN model (patient-wise split)")
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch: {epoch+1}, Train Loss:  {train_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "Results/compact_patient_split.pth")

all_preds, all_labels = evaluate(model, test_loader, device)

acc_score, f1 = compute_metrics(all_labels, all_preds)

print(f"Accuracy: {acc_score:.2f}")
print(f"F1 Score: {f1:.2f}")

results = {
    "accuracy": float(acc_score),
    "f1 score": float(f1)
}

with open("Results/compact_patient_split_results.json", "w") as f:
    json.dump(results, f, indent=4)

cm = get_confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Compact Patient-wise Split Confusion Matrix")
plt.savefig("Results/compact_confusion_matrix.png")
plt.close()

plot_confusion_matrix(cm)
