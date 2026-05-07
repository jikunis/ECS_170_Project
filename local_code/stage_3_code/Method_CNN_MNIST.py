'''
Concrete MethodModule class for CNN on MNIST dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # MNIST: 28x28, 1 channel, 10 classes
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.activation_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.activation_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # 64 channels * 7 * 7 = 3136
        self.fc_layer_1 = nn.Linear(64 * 7 * 7, 128)
        self.activation_3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.pool_1(self.activation_1(self.conv_layer_1(x)))
        x = self.pool_2(self.activation_2(self.conv_layer_2(x)))

        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.activation_3(self.fc_layer_1(x)))
        x = self.fc_layer_2(x)     # raw logits
        return x

    def train_model(self, X, y):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        print(f'Using device: {device}')

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # MNIST images are 28x28 with no channel dim — need to add it
        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1).to(device)  # (N, 1, 28, 28)
        y_tensor = torch.LongTensor(np.array(y)).to(device)

        n = X_tensor.size(0)
        self.loss_history = []

        for epoch in range(self.max_epoch):
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

            if epoch % 2 == 0:
                with torch.no_grad():
                    y_pred_all = self.forward(X_tensor)
                    accuracy_evaluator.data = {
                        'true_y': y_tensor.cpu(),
                        'pred_y': y_pred_all.max(1)[1].cpu()
                    }
                    print(f'Epoch: {epoch} | Accuracy: {accuracy_evaluator.evaluate()["accuracy"]:.4f} | Loss: {epoch_loss:.4f}')

    def test(self, X):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)

        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1).to(device)
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