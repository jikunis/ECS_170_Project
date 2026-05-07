'''
Concrete MethodModule class for CNN on ORL dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 32  # small batch size since ORL only has 360 training images

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # ORL: 112x92, 1 channel, 40 classes
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.activation_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x92 -> 56x46

        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.activation_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x46 -> 28x23

        # 64 * 28 * 23 = 41216
        self.fc_layer_1 = nn.Linear(64 * 28 * 23, 128)  # change 256 to 128
        self.activation_3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_layer_2 = nn.Linear(128, 40)  # match input to 128




    def forward(self, x):
        # x shape: (batch, 1, 112, 92)
        x = self.pool_1(self.activation_1(self.conv_layer_1(x)))
        x = self.pool_2(self.activation_2(self.conv_layer_2(x)))

        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.activation_3(self.fc_layer_1(x)))
        x = self.fc_layer_2(x)
        return x

    def train_model(self, X, y):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        print(f'Using device: {device}')

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # ORL images are (112, 92) with one channel extracted — add channel dim
        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1).to(device)  # (N, 1, 112, 92)
        # ORL labels are 1-40, shift to 0-39 for CrossEntropyLoss
        y_tensor = torch.LongTensor(np.array(y) - 1).to(device)

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

            if epoch % 5 == 0:
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
        # shift predictions back to 1-40
        return y_pred.max(1)[1].cpu() + 1

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}