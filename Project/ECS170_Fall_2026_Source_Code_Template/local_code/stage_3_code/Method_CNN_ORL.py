'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 15
    # it defines the size of each batch to train the model on
    batch_size = 32
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    eval_batch_size = 1000

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, in_channels=1, image_size=(112, 92), num_classes=40):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # convolutional feature extractor: (N,C,H,W) -> (N,64,H/4,W/4)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # figure out the flattened size with a dummy pass (3136 for MNIST)
        with torch.no_grad():
            flat_size = self.features(torch.zeros(1, in_channels, *image_size)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        return self.classifier(self.features(x))

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    # converts to a tensor
    def _to_tensor(self, X):
        X = torch.FloatTensor(np.array(X))
        if X.max() > 1.0:      # scale 0-255 pixels to 0-1
            X = X / 255.0
        return X

    # batched prediction so big datasets don't blow up memory
    def _predict(self, X_t):
        preds = []
        with torch.no_grad():
            for s in range(0, X_t.shape[0], self.eval_batch_size):
                preds.append(self.forward(X_t[s:s + self.eval_batch_size]).max(1)[1])
        return torch.cat(preds)

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        evaluator = Evaluate_Metrics('training evaluator', '')

        X_t = self._to_tensor(X)
        y_t = torch.LongTensor(np.array(y))
        X_test_t = self._to_tensor(self.data['test']['X'])
        n = X_t.shape[0]
        self.history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(self.max_epoch):
            perm = torch.randperm(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                logits = self.forward(X_t[idx])
                train_loss = loss_function(logits, y_t[idx])

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * len(idx)

            # per-epoch metrics for the convergence plot
            pred = self._predict(X_t)
            evaluator.data = {'true_y': y_t, 'pred_y': pred}
            metrics = evaluator.evaluate_all()

            test_pred = self._predict(X_test_t)
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
        return self._predict(self._to_tensor(X))

    def run(self):
        print('method running...')
        
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            
            