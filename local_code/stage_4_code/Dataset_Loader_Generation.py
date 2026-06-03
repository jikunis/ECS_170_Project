'''
Dataset Loader for text generation
Reads a single plain-text file and builds overlapping word-level sequences.
Each sample: input = seq_len tokens, target = next token.
'''

import re
from collections import Counter
from local_code.base_class.dataset import dataset


class Dataset_Loader_Generation(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    seq_len = 30          # context window length
    vocab_size = 5000     # keep top-N words
    vocab = None          # word -> index  (built during load)
    idx2word = None       # index -> word  (for generation)

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = text.split()
        return tokens

    def load(self):
        print('loading generation data...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()

        tokens = self._clean(raw)
        print(f'Total tokens after cleaning: {len(tokens)}')

        # build vocab
        counter = Counter(tokens)
        most_common = [w for w, _ in counter.most_common(self.vocab_size - 2)]
        # 0 = PAD, 1 = UNK
        self.vocab    = {w: i + 2 for i, w in enumerate(most_common)}
        self.idx2word = {i + 2: w for i, w in enumerate(most_common)}
        self.idx2word[0] = '<PAD>'
        self.idx2word[1] = '<UNK>'
        print(f'Vocabulary size: {len(self.vocab) + 2}')

        # encode all tokens
        ids = [self.vocab.get(t, 1) for t in tokens]

        # build (input_seq, target_word) pairs with stride 1
        X, y = [], []
        for i in range(len(ids) - self.seq_len):
            X.append(ids[i: i + self.seq_len])
            y.append(ids[i + self.seq_len])

        # 90 / 10 train-test split
        split = int(len(X) * 0.9)
        print(f'Train sequences: {split} | Test sequences: {len(X) - split}')

        return {
            'train': {'X': X[:split], 'y': y[:split]},
            'test':  {'X': X[split:], 'y': y[split:]},
            'vocab_size': len(self.vocab) + 2,
            'vocab':      self.vocab,
            'idx2word':   self.idx2word
        }
