'''
Concrete MethodModule for GCN Node Classification - Stage 5
Follows professor Jiawei Zhang's framework conventions (GResNet repo).
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# Student implementation for ECS 170 Stage 5

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score

from GraphConvolution import GraphConvolution


class MethodGCN(nn.Module):
    """
    Two-layer GCN: Z = softmax(A_hat * ReLU(A_hat * X * W0) * W1)
    where A_hat = D^{-1/2} (A + I) D^{-1/2}
    """

    data = None
    lr = 0.01
    weight_decay = 5e-4
    epoch = 200
    hidden = 64
    dropout = 0.5

    def __init__(self, nfeat, nhid, nclass, dropout):
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class Method_GCN:
    """
    Training + evaluation wrapper — mirrors professor's MethodGCN conventions.
    """

    data = None
    dataset_name = 'cora'
    max_epoch = 200
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_size = 64
    dropout = 0.5
    result_destination_folder_path = './result/stage_5_result'

    # ------------------------------------------------------------------ #

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    @staticmethod
    def _accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        return preds.eq(labels).float().sum().item() / len(labels)

    # ------------------------------------------------------------------ #

    def train(self):
        device = self._get_device()
        print(f'\n{"="*60}')
        print(f'  Dataset : {self.dataset_name.upper()}')
        print(f'  Device  : {device}')
        print(f'{"="*60}')

        # unpack
        graph     = self.data['graph']
        ttv       = self.data['train_test_val']
        features  = graph['X'].to(device)
        labels    = graph['y'].to(device)
        adj       = graph['utility']['A'].to(device)
        idx_train = ttv['idx_train'].to(device)
        idx_val   = ttv['idx_val'].to(device)
        idx_test  = ttv['idx_test'].to(device)

        nfeat  = features.shape[1]
        nclass = int(labels.max().item()) + 1

        model = MethodGCN(nfeat, self.hidden_size, nclass, self.dropout).to(device)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        print(f'  Features: {nfeat}  |  Classes: {nclass}  |  Hidden: {self.hidden_size}')
        print(f'  Train: {len(idx_train)}  |  Val: {len(idx_val)}  |  Test: {len(idx_test)}')
        print(f'  Epochs: {self.max_epoch}  |  LR: {self.learning_rate}  |  WD: {self.weight_decay}')
        print('-'*60)

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []

        t_start = time.time()

        for epoch in range(1, self.max_epoch + 1):
            # ---- train ----
            model.train()
            optimizer.zero_grad()
            out        = model(features, adj)
            loss_train = F.nll_loss(out[idx_train], labels[idx_train])
            acc_train  = self._accuracy(out[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            # ---- validate ----
            model.eval()
            with torch.no_grad():
                out      = model(features, adj)
                loss_val = F.nll_loss(out[idx_val], labels[idx_val])
                acc_val  = self._accuracy(out[idx_val], labels[idx_val])

            train_losses.append(loss_train.item())
            val_losses.append(loss_val.item())
            train_accs.append(acc_train)
            val_accs.append(acc_val)

            if epoch % 20 == 0 or epoch == 1:
                print(f'  Epoch {epoch:>3d}/{self.max_epoch} | '
                      f'Train Loss {loss_train.item():.4f} Acc {acc_train:.4f} | '
                      f'Val Loss {loss_val.item():.4f} Acc {acc_val:.4f}')

        elapsed = time.time() - t_start

        # ---- test ----
        model.eval()
        with torch.no_grad():
            out       = model(features, adj)
            loss_test = F.nll_loss(out[idx_test], labels[idx_test])
            acc_test  = self._accuracy(out[idx_test], labels[idx_test])

            preds_test = out[idx_test].max(1)[1].cpu().numpy()
            true_test  = labels[idx_test].cpu().numpy()

        precision = precision_score(true_test, preds_test, average='macro', zero_division=0)
        recall    = recall_score(true_test, preds_test, average='macro', zero_division=0)
        f1        = f1_score(true_test, preds_test, average='macro', zero_division=0)

        print('-'*60)
        print(f'  Training time : {elapsed:.1f}s')
        print(f'  Test Loss     : {loss_test.item():.4f}')
        print(f'  Test Accuracy : {acc_test:.4f}')
        print(f'  Precision     : {precision:.4f}')
        print(f'  Recall        : {recall:.4f}')
        print(f'  F1 Score      : {f1:.4f}')

        self._save_curves(train_losses, val_losses, train_accs, val_accs)
        return acc_test, loss_test.item(), precision, recall, f1

    # ------------------------------------------------------------------ #

    def _save_curves(self, train_losses, val_losses, train_accs, val_accs):
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        epochs = range(1, len(train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'GCN — {self.dataset_name}', fontsize=14, fontweight='bold')

        ax1.plot(epochs, train_losses, label='Train', color='steelblue')
        ax1.plot(epochs, val_losses,   label='Val',   color='tomato', linestyle='--')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('NLL Loss')
        ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, train_accs, label='Train', color='steelblue')
        ax2.plot(epochs, val_accs,   label='Val',   color='tomato', linestyle='--')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.result_destination_folder_path,
                                f'{self.dataset_name}_learning_curves.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Learning curves → {out_path}')