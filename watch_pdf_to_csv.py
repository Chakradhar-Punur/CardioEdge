import pymupdf as fitz
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

pdf_path = "/Users/Chakradhar/PycharmProjects/ECG_Project/watch_ecg.pdf"

doc = fitz.open(pdf_path)
page = doc.load_page(0)

pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
image_path = "watch_ecg.png"
pix.save(image_path)

img = cv.imread(image_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(gray, 180, 255, cv.THRESH_BINARY_INV)

points = []

height, width = thresh.shape

for x in range(width):
    ys = np.where(thresh[:, x] > 0)[0]

    if len(ys) > 0:
        y = np.mean(ys)
        points.append(y)

signal = np.array(points)

signal = -signal

signal = signal - np.mean(signal)
signal = signal / (np.std(signal) + 1e-8)

# Smooth extracted ECG signal
window = 5
signal = np.convolve(
    signal,
    np.ones(window) / window,
    mode='same'
)

signal = np.interp(
    np.linspace(0, len(signal)-1, 234),
    np.arange(len(signal)),
    signal
)

# Save CSV
np.savetxt("watch_ecg.csv", signal, delimiter=",")

# Plot extracted ECG
plt.figure(figsize=(12, 4))
plt.plot(signal, linewidth=2)
plt.title("Extracted Apple Watch ECG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

print("ECG CSV saved as watch_ecg.csv")
