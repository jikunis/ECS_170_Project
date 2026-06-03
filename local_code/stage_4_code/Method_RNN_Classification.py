'''
Concrete MethodModule class for RNN-based text classification (IMDB sentiment)
Supports rnn_type = 'RNN' | 'LSTM' | 'GRU'
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_RNN_Classification(method, nn.Module):
    data = None
    max_epoch    = 10
    learning_rate = 1e-3
    batch_size   = 64
    embed_dim    = 128
    hidden_dim   = 256
    num_layers   = 2
    dropout      = 0.5
    rnn_type     = 'LSTM'   # 'RNN' | 'LSTM' | 'GRU'
    vocab_size   = None     # set from loaded data

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # layers are built lazily in _build() once vocab_size is known

    def _build(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)       # binary: pos / neg
        )

    def forward(self, x):
        # x: (batch, seq_len) int tensor
        emb = self.embedding(x)                         # (batch, seq_len, embed_dim)

        if self.rnn_type == 'LSTM':
            out, (h, _) = self.rnn(emb)
        else:
            out, h = self.rnn(emb)

        # take last-layer hidden state from final time step
        last_hidden = h[-1]                             # (batch, hidden_dim)
        return self.fc(last_hidden)

    # ------------------------------------------------------------------
    def train_model(self, X, y):
        self.vocab_size = self.data['vocab_size']
        self._build()

        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print(f'[{self.rnn_type}] Using device: {device}')

        optimizer      = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn        = nn.CrossEntropyLoss()
        accuracy_eval  = Evaluate_Accuracy('training evaluator', '')

        X_t = torch.LongTensor(np.array(X)).to(device)
        y_t = torch.LongTensor(np.array(y)).to(device)
        n   = X_t.size(0)
        self.loss_history = []

        for epoch in range(self.max_epoch):
            self.train()
            indices    = torch.randperm(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                idx     = indices[start: start + self.batch_size]
                X_batch = X_t[idx]
                y_batch = y_t[idx]

                y_pred = self.forward(X_batch)
                loss   = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss)

            # eval every 2 epochs on a sample to keep logs readable
            if epoch % 2 == 0:
                self.eval()
                with torch.no_grad():
                    sample_size = min(2000, n)
                    preds = self.forward(X_t[:sample_size])
                    accuracy_eval.data = {
                        'true_y': y_t[:sample_size].cpu(),
                        'pred_y': preds.max(1)[1].cpu()
                    }
                    acc = accuracy_eval.evaluate()['accuracy']
                    print(f'Epoch: {epoch:3d} | Loss: {epoch_loss:.4f} | Train Acc (sample): {acc:.4f}')

    # ------------------------------------------------------------------
    def test(self, X):
        device = next(self.parameters()).device
        self.eval()
        X_t = torch.LongTensor(np.array(X)).to(device)
        all_preds = []
        with torch.no_grad():
            for start in range(0, X_t.size(0), self.batch_size):
                batch = X_t[start: start + self.batch_size]
                preds = self.forward(batch)
                all_preds.append(preds.max(1)[1].cpu())
        return torch.cat(all_preds)

    # ------------------------------------------------------------------
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}