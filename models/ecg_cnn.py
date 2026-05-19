import torch.nn as nn
import torch.nn.functional as f


class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 29, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool1(f.relu(self.conv1(x)))
        x = self.pool2(f.relu(self.conv2(x)))
        x = self.pool3(f.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x
