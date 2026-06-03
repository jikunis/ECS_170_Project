'''
Dataset Loader for text classification (IMDB sentiment)
Reads from directory structure: train/pos, train/neg, test/pos, test/neg
'''

import os
import re
import string
from collections import Counter
from local_code.base_class.dataset import dataset


class Dataset_Loader_Classification(dataset):
    data = None
    dataset_source_folder_path = None
    vocab_size = 10000      # keep top-N words
    max_seq_len = 200       # truncate / pad to this length
    vocab = None            # word -> index, built during load()

    # common stop words to strip
    STOP_WORDS = {
        'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
        'yourself','yourselves','he','him','his','himself','she','her','hers',
        'herself','it','its','itself','they','them','their','theirs','themselves',
        'what','which','who','whom','this','that','these','those','am','is','are',
        'was','were','be','been','being','have','has','had','having','do','does',
        'did','doing','a','an','the','and','but','if','or','because','as','until',
        'while','of','at','by','for','with','about','against','between','into',
        'through','during','before','after','above','below','to','from','up','down',
        'in','out','on','off','over','under','again','further','then','once','here',
        'there','when','where','why','how','all','both','each','few','more','most',
        'other','some','such','no','nor','not','only','own','same','so','than',
        'too','very','s','t','can','will','just','don','should','now','d','ll',
        'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
        'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
        'weren','won','wouldn'
    }

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # ------------------------------------------------------------------
    # text cleaning
    # ------------------------------------------------------------------
    def _clean(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML tags
        text = re.sub(r'[^a-z\s]', ' ', text)         # keep only letters
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 1]
        return tokens

    # ------------------------------------------------------------------
    # read all .txt files from one directory, return list of token lists
    # ------------------------------------------------------------------
    def _read_dir(self, path):
        docs = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith('.txt'):
                with open(os.path.join(path, fname), 'r', encoding='utf-8', errors='replace') as f:
                    docs.append(self._clean(f.read()))
        return docs

    # ------------------------------------------------------------------
    # encode a token list to a fixed-length integer sequence
    # ------------------------------------------------------------------
    def _encode(self, tokens):
        PAD, UNK = 0, 1
        ids = [self.vocab.get(t, UNK) for t in tokens[:self.max_seq_len]]
        ids += [PAD] * (self.max_seq_len - len(ids))
        return ids

    # ------------------------------------------------------------------
    # main load
    # ------------------------------------------------------------------
    def load(self):
        print('loading classification data...')
        base = self.dataset_source_folder_path

        # --- read raw tokens ---
        train_pos = self._read_dir(os.path.join(base, 'train', 'pos'))
        train_neg = self._read_dir(os.path.join(base, 'train', 'neg'))
        test_pos  = self._read_dir(os.path.join(base, 'test',  'pos'))
        test_neg  = self._read_dir(os.path.join(base, 'test',  'neg'))

        train_docs = train_pos + train_neg
        train_labels = [1] * len(train_pos) + [0] * len(train_neg)
        test_docs  = test_pos  + test_neg
        test_labels  = [1] * len(test_pos)  + [0] * len(test_neg)

        # --- build vocabulary from training data only ---
        counter = Counter()
        for tokens in train_docs:
            counter.update(tokens)
        # index 0 = PAD, 1 = UNK, 2+ = real words
        most_common = [w for w, _ in counter.most_common(self.vocab_size - 2)]
        self.vocab = {w: i + 2 for i, w in enumerate(most_common)}
        print(f'Vocabulary size: {len(self.vocab) + 2} (including PAD/UNK)')

        # --- encode ---
        X_train = [self._encode(d) for d in train_docs]
        X_test  = [self._encode(d) for d in test_docs]

        print(f'Train: {len(X_train)} | Test: {len(X_test)}')
        return {
            'train': {'X': X_train, 'y': train_labels},
            'test':  {'X': X_test,  'y': test_labels},
            'vocab_size': len(self.vocab) + 2
        }
