'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 30
    # it defines the size of each batch to train the model on
    batch_size = 128
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.Tanh()

        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.Tanh()

        self.fc_layer_3 = nn.Linear(128, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))

        # outout layer result
        return self.fc_layer_3(h)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    # converts to a tensor
    def _to_tensor(self, X):
            X = torch.FloatTensor(np.array(X))
            if X.max() > 1.0:      # scale 0-255 pixels to 0-1
                X = X / 255.0
            return X

    def train(self, X, y):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        loss_function = nn.CrossEntropyLoss()

        # for training accuracy investigation purpose
        evaluator = Evaluate_Metrics('training evaluator', '')

        X_t = self._to_tensor(X)
        y_t = torch.LongTensor(np.array(y))
        n = X_t.shape[0]
        self.history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):

            perm = torch.randperm(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                logits = self.forward(X_t[idx])

                # calculate the training loss
                train_loss = loss_function(logits, y_t[idx])

                optimizer.zero_grad()

                # do the error backpropagation to calculate the gradients
                train_loss.backward()

                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                epoch_loss += train_loss.item() * len(idx)

            with torch.no_grad():
                pred = self.forward(X_t).max(1)[1]
                evaluator.data = {'true_y': y_t, 'pred_y': pred}
                metrics = evaluator.evaluate_all()

                # test accuracy each epoch, for the convergence plot
                test_pred = self.test(self.data['test']['X'])
                test_acc = accuracy_score(self.data['test']['y'], test_pred)

                # save epochs, loss and accuracy in history
                # used to plot on the graphs later
                self.history['epoch'].append(epoch + 1)
                self.history['loss'].append(epoch_loss / n)
                self.history['train_acc'].append(metrics['accuracy'])
                self.history['test_acc'].append(test_acc)

                # output metrics for this run
                print('Epoch:', epoch)
                print(f"Accuracy: {metrics['accuracy']:.4f} Precision: {metrics['precision']:.4f} "
                        f"Recall: {metrics['recall']:.4f} F1: {metrics['f1']:.4f} ")
                print('Loss:', epoch_loss / n)

    
    def test(self, X):
        # do the testing, and return the result
        with torch.no_grad():
            logits = self.forward(self._to_tensor(X))
        return logits.max(1)[1]

    # run the train and test functions defined above
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            
            