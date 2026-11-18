'''
Concrete IO class for the IMDb movie review dataset (binary sentiment)
'''

import os
import re
import string
from collections import Counter

import numpy as np

from local_code.base_class.dataset import dataset


# Fallback stop word list, used if NLTK's list isn't available offline
_FALLBACK_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could',
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
    'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my',
    'myself', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours',
    'yourself', 'yourselves',
}

# Negations carry sentiment ("not good" vs "good"), so never drop them
_NEGATIONS = {
    'no', 'nor', 'not', 'never', 'neither', 'dont', 'didnt', 'doesnt', 'isnt', 'wasnt', 'wont',
    'cant', 'couldnt', 'wouldnt', 'shouldnt', 'arent', 'werent', 'hasnt', 'havent', 'hadnt',
    'aint', 'mustnt', 'neednt', 'shant',
}


def _build_stopwords():
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words('english'))
    except Exception:
        words = set(_FALLBACK_STOPWORDS)
    words = {w.replace("'", '') for w in words}   # "don't" -> "dont" (matches cleaning below)
    return words - _NEGATIONS


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None   # path to the aclImdb/ folder (contains train/ and test/)
    dataset_source_file_name = None     # unused, kept for interface compatibility
    dataset_name = 'IMDB'

    # text processing settings
    max_len = 400            # sequences are truncated / padded to this length
    max_vocab_size = 20000   # including PAD and UNK
    min_freq = 2             # drop words seen fewer times than this in the training set
    use_stemming = False     # Porter stemming (needs nltk)
    remove_stopwords = True

    PAD_TOKEN = '<PAD>'
    PAD_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_IDX = 1

    vocab = None             # word -> index, filled in by load()

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self._stopwords = _build_stopwords() if self.remove_stopwords else set()
        self._stemmer = None
        if self.use_stemming:
            from nltk.stem.porter import PorterStemmer
            self._stemmer = PorterStemmer()
        # strip apostrophes entirely, turn every other punctuation mark into a space
        self._punct_table = str.maketrans(
            {**{c: ' ' for c in string.punctuation}, "'": ''}
        )

    # ---------- text cleaning ----------
    def _clean(self, text):
        text = re.sub(r'<[^>]+>', ' ', text)         # HTML tags such as <br />
        text = text.lower()
        text = text.translate(self._punct_table)     # remove punctuation
        tokens = text.split()
        tokens = [t for t in tokens if t.isalpha() and len(t) > 1]   # drop numbers / noise
        if self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]
        return tokens

    # ---------- reading raw files ----------
    def _read_split(self, split):
        """Returns (list of token lists, np.array of labels) for 'train' or 'test'."""
        docs, labels = [], []
        for label_name, label in (('neg', 0), ('pos', 1)):
            folder = os.path.join(self.dataset_source_folder_path, split, label_name)
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                    docs.append(self._clean(f.read()))
                labels.append(label)
        return docs, np.array(labels, dtype=np.int64)

    # ---------- vocabulary ----------
    def _build_vocab(self, train_docs):
        # built from the training set only, so no test information leaks in
        counts = Counter(tok for doc in train_docs for tok in doc)
        words = [w for w, c in counts.most_common() if c >= self.min_freq]
        words = words[: self.max_vocab_size - 2]
        vocab = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        for w in words:
            vocab[w] = len(vocab)
        return vocab

    # ---------- tokens -> padded index matrix ----------
    def _encode(self, docs):
        N = len(docs)
        X = np.full((N, self.max_len), self.PAD_IDX, dtype=np.int64)
        lengths = np.zeros(N, dtype=np.int64)
        for i, doc in enumerate(docs):
            ids = [self.vocab.get(t, self.UNK_IDX) for t in doc[: self.max_len]]
            if len(ids) == 0:                  # guard: review became empty after cleaning
                ids = [self.UNK_IDX]
            X[i, : len(ids)] = ids
            lengths[i] = len(ids)
        return X, lengths

    def load(self):
        print('loading data...')

        train_docs, y_train = self._read_split('train')
        test_docs, y_test = self._read_split('test')

        self.vocab = self._build_vocab(train_docs)
        print(f'vocabulary size: {len(self.vocab)}')

        X_train, len_train = self._encode(train_docs)
        X_test, len_test = self._encode(test_docs)

        return {
            'train': {'X': X_train, 'y': y_train, 'lengths': len_train},
            'test': {'X': X_test, 'y': y_test, 'lengths': len_test},
        }