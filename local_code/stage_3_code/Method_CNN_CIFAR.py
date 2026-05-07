'''
Concrete MethodModule class for CNN on CIFAR-10 dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 64

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.network = nn.Sequential(
            # block 1: (3, 32, 32) -> (32, 16, 16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # block 2: (32, 16, 16) -> (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # block 3: (64, 8, 8) -> (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.contiguous().reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_model(self, X, y):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        print(f'Using device: {device}')

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # preprocess in numpy before sending to device
        X_np = np.array(X, dtype=np.float32) / 255.0          # normalize 0-1
        X_np = np.transpose(X_np, (0, 3, 1, 2))               # (N, H, W, C) -> (N, C, H, W)
        X_tensor = torch.tensor(X_np).to(device)
        y_tensor = torch.tensor(np.array(y, dtype=np.int64)).to(device)

        n = X_tensor.size(0)
        self.loss_history = []

        for epoch in range(self.max_epoch):
            self.train()
            indices = torch.randperm(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss)

            if epoch % 5 == 0:
                self.eval()
                with torch.no_grad():
                    # only evaluate on first 5000 samples to save memory
                    y_pred_sample = self.forward(X_tensor[:5000])
                    accuracy_evaluator.data = {
                        'true_y': y_tensor[:5000].cpu(),
                        'pred_y': y_pred_sample.max(1)[1].cpu()
                    }
                    print(
                        f'Epoch: {epoch} | Accuracy: {accuracy_evaluator.evaluate()["accuracy"]:.4f} | Loss: {epoch_loss:.4f}')
    def test(self, X):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        self.eval()

        X_np = np.array(X, dtype=np.float32) / 255.0
        X_np = np.transpose(X_np, (0, 3, 1, 2))
        X_tensor = torch.tensor(X_np).to(device)

        with torch.no_grad():
            y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1].cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}