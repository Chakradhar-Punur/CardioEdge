from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc_score, f1


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(7, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Confusion Matrix")
    plt.show()
