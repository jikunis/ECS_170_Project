'''
Stage 5 — Main Script
Trains a 2-layer GCN on Cora, Citeseer, and Pubmed.

Run from your project root:
    python local_code/stage_5_code/script_gcn.py
'''

import sys, os

# ---------- path setup ----------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # go up twice
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from Dataset_Loader_Node_Classification import Dataset_Loader
from Method_GCN import Method_GCN

# ---------- config ----------
DATA_ROOT  = os.path.join(project_root, 'data', 'stage_5_data', 'stage_5_data')
RESULT_DIR = os.path.join(project_root, 'result', 'stage_5_result')

DATASETS = ['cora', 'citeseer', 'pubmed']

HP = {
    'cora':     dict(hidden=32, dropout=0.5, lr=0.01, wd=5e-4, epochs=200),
    'citeseer': dict(hidden=32, dropout=0.5, lr=0.01, wd=5e-4, epochs=200),
    'pubmed':   dict(hidden=32, dropout=0.5, lr=0.01, wd=5e-4, epochs=200),
}

# ---------- run ----------
summary = {}

for name in DATASETS:
    data_path = os.path.join(DATA_ROOT, name)
    if not os.path.isdir(data_path):
        print(f'\n[SKIP] {name} — data not found at {data_path}')
        continue

    loader = Dataset_Loader(dName=name)
    loader.dataset_source_folder_path = data_path
    loaded = loader.load()

    hp = HP[name]
    method = Method_GCN()
    method.data          = loaded
    method.dataset_name  = name
    method.max_epoch     = hp['epochs']
    method.learning_rate = hp['lr']
    method.weight_decay  = hp['wd']
    method.hidden_size   = hp['hidden']
    method.dropout       = hp['dropout']
    method.result_destination_folder_path = RESULT_DIR

    acc, loss, prec, rec, f1 = method.train()
    summary[name] = {'accuracy': acc, 'loss': loss,
                     'precision': prec, 'recall': rec, 'f1': f1}

# ---------- summary ----------
print('\n' + '='*72)
print('  FINAL SUMMARY')
print('='*72)
print(f'  {"Dataset":<12}  {"Acc":>8}  {"Loss":>8}  {"Precision":>10}  {"Recall":>8}  {"F1":>8}')
print('  ' + '-'*60)
for name, r in summary.items():
    print(f'  {name:<12}  {r["accuracy"]:>8.4f}  {r["loss"]:>8.4f}  '
          f'{r["precision"]:>10.4f}  {r["recall"]:>8.4f}  {r["f1"]:>8.4f}')
print('='*72)
print(f'\nLearning curves saved to: {RESULT_DIR}')