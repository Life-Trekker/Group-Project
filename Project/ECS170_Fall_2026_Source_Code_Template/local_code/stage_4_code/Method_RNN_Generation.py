'''
Word-level RNN (LSTM/GRU/RNN) for text generation on the jokes dataset
'''

import csv
import math
import random
import re
from collections import Counter

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------- data
def tokenize(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\.{2,}', ' ... ', text)
    return re.findall(r"\.\.\.|[a-z0-9']+|[.,!?;:]", text.lower())


class TextGenData:
    def __init__(self, path, seq_len=20, min_freq=2, val_frac=0.1, seed=2):
        with open(path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            next(reader)                                       # skip "ID","Joke" header
            jokes = [row[1].strip() for row in reader if len(row) > 1 and row[1].strip()]

        jokes = list(dict.fromkeys(jokes))                     # remove exact duplicate jokes
        random.Random(seed).shuffle(jokes)                     # split by joke, not mid-joke
        cut = int(len(jokes) * (1 - val_frac))
        train_jokes, val_jokes = jokes[:cut], jokes[cut:]

        def to_tokens(js):
            toks = []
            for j in js:
                toks += tokenize(j) + ['<eos>']                # <eos> marks the end of a joke
            return toks

        train_tokens, val_tokens = to_tokens(train_jokes), to_tokens(val_jokes)

        # vocabulary from the training split only
        counts = Counter(train_tokens)
        words = sorted(w for w, c in counts.items() if c >= min_freq and w != '<unk>')
        self.itos = ['<unk>'] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        enc = lambda toks: np.array([self.stoi.get(t, 0) for t in toks], dtype=np.int64)
        self.train_ids, self.val_ids = enc(train_tokens), enc(val_tokens)
        self.ids = np.concatenate([self.train_ids, self.val_ids])
        self.tokens = train_tokens                             # reference text for comparison
        self.num_jokes = (len(train_jokes), len(val_jokes))
        self.seq_len = seq_len

    def windows(self, ids, stride=None):
        '''Input windows and the same windows shifted by one word (next-word targets).'''
        L = self.seq_len
        stride = stride or L
        starts = range(0, len(ids) - L, stride)
        x = np.stack([ids[s:s + L] for s in starts])
        y = np.stack([ids[s + 1:s + L + 1] for s in starts])
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------- model
class Method_RNN_Generation(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2,
                 rnn_type='LSTM', dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[rnn_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.history = None

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(self.drop(self.embedding(x)), hidden)
        return self.fc(self.drop(out)), hidden

    def fit(self, data, epochs=20, batch_size=64, lr=2e-3, clip=5.0, device='cpu'):
        self.to(device)
        # overlapping windows for training, non-overlapping for validation
        xtr, ytr = data.windows(data.train_ids, stride=max(1, data.seq_len // 2))
        xva, yva = data.windows(data.val_ids)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        history = {'epoch': [], 'loss': [], 'val_loss': [], 'val_ppl': []}
        best_val, best_state = float('inf'), None

        for epoch in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(len(xtr))
            total = 0.0
            for s in range(0, len(xtr), batch_size):
                idx = perm[s:s + batch_size]
                xb, yb = xtr[idx].to(device), ytr[idx].to(device)
                logits, _ = self(xb)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                opt.step()
                total += loss.item() * len(idx)

            self.eval()
            with torch.no_grad():
                vl, n = 0.0, 0
                for s in range(0, len(xva), 256):
                    xb, yb = xva[s:s + 256].to(device), yva[s:s + 256].to(device)
                    logits, _ = self(xb)
                    vl += loss_fn(logits.reshape(-1, logits.size(-1)),
                                  yb.reshape(-1)).item() * len(xb)
                    n += len(xb)
            train_loss, val_loss = total / len(xtr), vl / max(n, 1)
            history['epoch'].append(epoch)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(math.exp(val_loss))

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                history['best_epoch'] = epoch

            print(f'Epoch {epoch}: loss={train_loss:.4f} val_loss={val_loss:.4f} '
                  f'val_perplexity={math.exp(val_loss):.1f}')

        if best_state is not None:
            self.load_state_dict(best_state)
        print(f'Best epoch: {history["best_epoch"]} (val perplexity {math.exp(best_val):.1f})')

        self.history = history
        return history

    def generate(self, data, seed_words, max_words=40, temperature=0.7,
                 top_k=10, min_words=8, device='cpu'):
        self.eval()
        punct = {'.', ',', '!', '?', ';', ':', '...'}
        eos = data.stoi['<eos>']
        words = [w.lower() for w in seed_words]
        ids = torch.tensor([[data.stoi.get(w, 0) for w in words]], device=device)
        with torch.no_grad():
            logits, hidden = self(ids)
            for step in range(max_words):
                l = logits[0, -1].clone()
                l[0] = float('-inf')                              # never <unk>
                l[data.stoi.get('...', 0)] = float('-inf')        # never "..."
                if step < min_words:
                    l[eos] = float('-inf')                        # don't stop too early
                if words[-1] in punct:                            # no ". ." or "? ,"
                    for p in punct:
                        if p in data.stoi:
                            l[data.stoi[p]] = float('-inf')
                elif words[-1] in data.stoi:
                    l[data.stoi[words[-1]]] = float('-inf')       # no immediate word repeat
                l = l / temperature
                vals, idx = torch.topk(l, top_k)
                nxt = idx[torch.multinomial(torch.softmax(vals, dim=-1), 1)].item()
                if nxt == eos:
                    break
                words.append(data.itos[nxt])
                logits, hidden = self(torch.tensor([[nxt]], device=device), hidden)
        if words[-1] not in {'.', '?', '!'}:
            words.append('.')
        if words[-1] == '?' and words.count('?') > 1: 
            words[-1] = '.'
        return words


def detokenize(words):
    text = ''
    for w in words:
        text += w if w in {'.', ',', '!', '?', ';', ':'} else (' ' if text else '') + w
    text = re.sub(r"\bi\b", "I", text)
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------- evaluation
def ngram_overlap(generated, train_tokens, ns=(1, 2, 3, 4)):
    '''Fraction of the generated n-grams that also appear in the training text.'''
    result = {}
    for n in ns:
        train_set = set(zip(*[train_tokens[i:] for i in range(n)]))
        gen = list(zip(*[generated[i:] for i in range(n)]))
        result[n] = sum(g in train_set for g in gen) / max(len(gen), 1)
    return result


def longest_copied_span(generated, train_tokens):
    '''Length of the longest run of generated words that appears verbatim in the training text.'''
    best = 0
    for n in range(1, len(generated) + 1):
        train_set = set(zip(*[train_tokens[i:] for i in range(n)]))
        gen = zip(*[generated[i:] for i in range(n)])
        if any(g in train_set for g in gen):
            best = n
        else:
            break
    return best