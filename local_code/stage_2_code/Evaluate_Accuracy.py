'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        if hasattr(pred_y, 'detach'):
            pred_y = pred_y.detach().numpy()
        if hasattr(true_y, 'detach'):
            true_y = true_y.detach().numpy()

        return {
            'accuracy': accuracy_score(true_y, pred_y),
            'f1': f1_score(true_y, pred_y, average='weighted'),
            'precision': precision_score(true_y, pred_y, average='weighted'),
            'recall': recall_score(true_y, pred_y, average='weighted')
        }