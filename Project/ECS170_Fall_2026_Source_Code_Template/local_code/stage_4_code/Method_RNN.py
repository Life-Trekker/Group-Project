'''
RNN method for IMDb sentiment classification
'''

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


def load_glove_matrix(glove_path, vocab, dim):
    '''Builds a (vocab_size, dim) matrix; words not in GloVe get small random vectors, PAD gets zeros.'''
    matrix = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0                                   # PAD_IDX = 0
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, _, rest = line.partition(' ')
            if word in vocab:                         # only keep words we need
                vec = np.asarray(rest.split(), dtype=np.float32)
                if len(vec) == dim:
                    matrix[vocab[word]] = vec
                    found += 1
    print(f'GloVe coverage: {found}/{len(vocab)} vocabulary words')
    return torch.from_numpy(matrix)


class CustomRNN(nn.Module):
    '''Hand-written Elman RNN: h_t = tanh(W_x x_t + W_h h_{t-1} + b)'''

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_x = nn.Linear(input_dim, hidden_dim)                 # includes bias b
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        for t in range(T):
            h_new = torch.tanh(self.W_x(x[:, t]) + self.W_h(h))
            mask = (t < lengths).unsqueeze(1)    # padded steps keep the old hidden state
            h = torch.where(mask, h_new, h)
        return h                                 # hidden state at the last real token


class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 10
    batch_size = 64
    learning_rate = 1e-3
    eval_batch_size = 500
    clip_grad = 5.0

    def __init__(self, mName, mDescription, vocab_size, embed_dim=128, hidden_dim=128,
                 rnn_type='LSTM', bidirectional=True, dropout=0.5, num_classes=2,
                 pretrained_embeddings=None, freeze_embeddings=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional and rnn_type != 'CUSTOM'   # custom cell is one-directional

        # embedding (optionally initialised from GloVe; embed_dim must match the GloVe dimension)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        if rnn_type == 'CUSTOM':
            self.rnn = CustomRNN(embed_dim, hidden_dim)
        else:
            rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[rnn_type]
            self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=self.bidirectional)

        feat_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, lengths):
        '''Forward propagation'''
        emb = self.drop(self.embedding(x))

        if self.rnn_type == 'CUSTOM':
            h = self.rnn(emb, lengths)
        else:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            _, h = self.rnn(packed)
            if self.rnn_type == 'LSTM':
                h = h[0]                                    # (layers*dirs, B, H)
            if self.bidirectional:
                h = torch.cat([h[-2], h[-1]], dim=1)        # forward + backward
            else:
                h = h[-1]

        return self.classifier(self.drop(h))

    @staticmethod
    def _get_lengths(split):
        '''True review lengths: use the loader's values if present, else count non-PAD tokens.'''
        if 'lengths' in split:
            return torch.LongTensor(np.array(split['lengths']))
        X = np.array(split['X'])
        return torch.LongTensor(np.maximum((X != 0).sum(axis=1), 1))   # PAD_IDX = 0

    # NOTE: this class overrides nn.Module.train(), so nn.Module.eval() (which calls
    # self.train(False)) would break. Switch modes through nn.Module.train explicitly.
    def _set_mode(self, training):
        nn.Module.train(self, training)

    def _predict(self, X_t, len_t):
        self._set_mode(False)
        preds = []
        with torch.no_grad():
            for s in range(0, X_t.shape[0], self.eval_batch_size):
                xb = X_t[s:s + self.eval_batch_size]
                lb = len_t[s:s + self.eval_batch_size]
                if self.rnn_type == 'CUSTOM':               # no need to loop over all-padding steps
                    xb = xb[:, :int(lb.max())]
                preds.append(self.forward(xb, lb).max(1)[1])
        return torch.cat(preds)

    def train(self, X, y):
        # only pass trainable parameters (matters when the embeddings are frozen)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        evaluator = Evaluate_Metrics('training evaluator', '')

        X_t = torch.LongTensor(np.array(X))
        len_t = self._get_lengths(self.data['train'])
        y_t = torch.LongTensor(np.array(y))
        X_test_t = torch.LongTensor(np.array(self.data['test']['X']))
        len_test_t = self._get_lengths(self.data['test'])
        n = X_t.shape[0]
        self.history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(self.max_epoch):
            self._set_mode(True)
            perm = torch.randperm(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, lb = X_t[idx], len_t[idx]
                if self.rnn_type == 'CUSTOM':
                    xb = xb[:, :int(lb.max())]
                logits = self.forward(xb, lb)
                train_loss = loss_function(logits, y_t[idx])

                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
                optimizer.step()

                epoch_loss += train_loss.item() * len(idx)

            pred = self._predict(X_t, len_t)
            evaluator.data = {'true_y': y_t, 'pred_y': pred}
            metrics = evaluator.evaluate_all()

            test_pred = self._predict(X_test_t, len_test_t)
            test_acc = accuracy_score(self.data['test']['y'], test_pred)

            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss / n)
            self.history['train_acc'].append(metrics['accuracy'])
            self.history['test_acc'].append(test_acc)

            print('Epoch:', epoch + 1)
            print(f"Accuracy: {metrics['accuracy']:.4f} Precision: {metrics['precision']:.4f} "
                  f"Recall: {metrics['recall']:.4f} F1: {metrics['f1']:.4f}")
            print('Loss:', epoch_loss / n)

    def test(self, X):
        X_t = torch.LongTensor(np.array(X))
        len_t = self._get_lengths(self.data['test'])
        return self._predict(X_t, len_t)

    def run(self):
        print('method running...')

        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}