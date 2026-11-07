'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# used to be Evaluate_Accuracy
# changed so now it evaluates precision, recall and f1 scores as well
class Evaluate_Metrics(evaluate):
    data = None
    average = 'macro'
    
    def evaluate_accuracy(self):
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def evaluate_f1(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], average=self.average, zero_division=0)

    def evaluate_precision(self):
        return precision_score(self.data['true_y'], self.data['pred_y'], average=self.average, zero_division=0)

    def evaluate_recall(self):
        return recall_score(self.data['true_y'], self.data['pred_y'], average=self.average, zero_division=0)

    def evaluate_all(self):
        # print('evaluating all metrics...')
        return {
            'accuracy':  self.evaluate_accuracy(),
            'precision': self.evaluate_precision(),
            'recall':    self.evaluate_recall(),
            'f1':        self.evaluate_f1(),
        }

    # evaluates only accuracy by default
    def evaluate(self):
        return self.evaluate_accuracy()
        