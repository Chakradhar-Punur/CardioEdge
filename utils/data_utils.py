import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():
    X = np.load("/Users/Chakradhar/PycharmProjects/ECG_Project/X.npy")
    y = np.load("/Users/Chakradhar/PycharmProjects/ECG_Project/y.npy")
    patient_ids = np.load("/Users/Chakradhar/PycharmProjects/ECG_Project/patient_ids.npy")

    le = LabelEncoder()
    y = le.fit_transform(y)

    unique_patients = np.unique(patient_ids)

    np.random.seed(42)
    np.random.shuffle(unique_patients)

    n = len(unique_patients)

    train_patients = unique_patients[:int(0.7 * n)]
    val_patients = unique_patients[int(0.7 * n):int(0.85 * n)]
    test_patients = unique_patients[int(0.85 * n):]

    train_mask = np.isin(patient_ids, train_patients)
    val_mask = np.isin(patient_ids, val_patients)
    test_mask = np.isin(patient_ids, test_patients)

    X_train, y_train = X[train_mask], y[train_mask]

    X_val, y_val = X[val_mask], y[val_mask]

    X_test, y_test = X[test_mask], y[test_mask]

    print("Train patients:", len(train_patients))
    print("Val patients:", len(val_patients))
    print("Test patients:", len(test_patients))
    print("Train samples:", len(X_train))
    print("Val samples:", len(X_val))
    print("Test samples:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    return X, y
