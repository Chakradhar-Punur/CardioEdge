import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

VALID_LABELS = set(AAMI_MAP.keys())

dataset_path = os.path.join(os.getcwd(), "data", "mit-bih-arrhythmia-database-1.0.0")
record = wfdb.rdrecord(os.path.join(dataset_path, "100"))

print("Sampling frequency: ", record.fs)
print("Signal Shape: ", record.p_signal.shape)
print("Signal channels:", record.sig_name)

records = sorted(list(set([f.split('.')[0] for f in os.listdir(dataset_path) if '.dat' in f])))

all_beats = []
all_beat_labels = []
all_patient_ids = []

for rec in records:
    patient_id = rec
    record_path = os.path.join(dataset_path, rec)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]
    symbols = annotation.symbol
    r_peaks = annotation.sample

    pre_samples = 90
    post_samples = 144

    for peak, symbol in zip(r_peaks, symbols):
        if symbol not in VALID_LABELS:
            continue

        start = peak - pre_samples
        end = peak + post_samples

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        if beat.std() != 0:
            beat = (beat - beat.mean()) / beat.std()

        all_beats.append(beat)

        all_beat_labels.append(AAMI_MAP[symbol])
        all_patient_ids.append(patient_id)

print("Extracted beats: ", len(all_beats))
print("Length of each beat: ", len(all_beats[0]) if all_beats else 0)

print("Class Distributions: ")
print(Counter(all_beat_labels))

plot_record_path = os.path.join(dataset_path, "100")
plot_record = wfdb.rdrecord(plot_record_path)
plot_annotation = wfdb.rdann(plot_record_path, 'atr')

plot_signal = plot_record.p_signal[:, 0]
plot_r_peaks = plot_annotation.sample
plot_symbols = plot_annotation.symbol

plt.figure(figsize=(10, 4))
plt.plot(plot_signal[:1000], label="ECG_Signal")

for peak in plot_r_peaks:
    if peak < 1000:
        plt.scatter(peak, plot_signal[peak], color='red', s=15)

plt.title("ECG Signal with R-peak annotations")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

X = np.array(all_beats)
y = np.array(all_beat_labels)
patient_ids = np.array(all_patient_ids)
print("Final dataset shape: ", X.shape, y.shape)

np.save("X.npy", X)
np.save("y.npy", y)
np.save("patient_ids.npy", patient_ids)
