""" Provides ClassificationMeasures. """

from typing import Literal

import numpy as np
from numpy.typing import NDArray


class ClassificationMeasures:
    """ Computes common evaluation measures for classification based tasks. """

    def __init__(self, y_true: NDArray, y_pred: NDArray):
        """ Initializes the class to compute on given data.

        Args:
            y_true: Array containing true values.
            y_pred: Array containing predicted values.
        """

        # Store the passed arguments
        self.y_true = y_true
        self.y_pred = y_pred

        # Initialize the classes based on y values
        self.classes = np.unique(np.concatenate((y_true, y_pred)))
        self.num_classes = self.classes.shape[0]

        # Initialize the confusion matrices to None
        self.confusion_matrices = None


    def accuracy_score(self) -> float:
        """ Computes the accuracy. """

        return np.mean(self.y_true == self.y_pred)


    def compute_confusion_matrices(self) -> NDArray:
        """ Compute the confusion matrices for each class. """

        confusion_matrices = np.empty((self.num_classes, 2, 2))

        # Fill the confusion matrix for each classes
        for idx, clx in enumerate(self.classes):
            # True positive
            confusion_matrices[idx, 0, 0] = np.sum((self.y_true == clx) & (self.y_pred == clx))
            # False positive
            confusion_matrices[idx, 0, 1] = np.sum((self.y_true != clx) & (self.y_pred == clx))
            # False negative
            confusion_matrices[idx, 1, 0] = np.sum((self.y_true == clx) & (self.y_pred != clx))
            # True negative
            confusion_matrices[idx, 1, 1] = np.sum((self.y_true != clx) & (self.y_pred != clx))

        return confusion_matrices


    def f1_score(self, average: Literal['micro', 'macro']) -> float:
        """ Computes the f1 score. """

        # Validate the passed arguments
        assert average in ['micro', 'macro'], f'Unrecognized argument for average {average}'

        # Compute recall and precision with same method
        recall = self.recall_score(average)
        precision = self.precision_score(average)

        # Compute the F1 score
        f1 = 2 * recall * precision / (recall + precision)
        return f1


    def recall_score(self, average: Literal['micro', 'macro']) -> float:
        """ Computes the recall. """

        # Validate the passed arguments
        assert average in ['micro', 'macro'], f'Unrecognized argument for average {average}'

        # Compute confusion matrix for each class
        if self.confusion_matrices is None:
            self.confusion_matrices = self.compute_confusion_matrices()

        if average == 'micro':
            # Compute recall of pooled confusion matrix
            pooled_confusion_matrix = np.sum(self.confusion_matrices, axis=0)
            recall = pooled_confusion_matrix[0, 0] / \
                            (pooled_confusion_matrix[0, 0] + pooled_confusion_matrix[0, 1])

        elif average == 'macro':
            # Compute average over recall of individual classes
            recall = 0
            for idx in range(self.num_classes):
                denom = self.confusion_matrices[idx, 0, 0] + self.confusion_matrices[idx, 0, 1]
                if denom != 0:
                    recall += (self.confusion_matrices[idx, 0, 0] / denom)
                else:
                    recall += 1
            recall /= self.num_classes

        return recall


    def precision_score(self, average: Literal['micro', 'macro']) -> float:
        """ Computes the precision. """

        # Validate the passed arguments
        assert average in ['micro', 'macro'], f'Unrecognized argument for average {average}'

        # Compute confusion matrix for each class
        if self.confusion_matrices is None:
            self.confusion_matrices = self.compute_confusion_matrices()

        if average == 'micro':
            # Compute precision of pooled confusion matrix
            pooled_confusion_matrix = np.sum(self.confusion_matrices, axis=0)
            precision = pooled_confusion_matrix[0, 0] / \
                                (pooled_confusion_matrix[0, 0] + pooled_confusion_matrix[1, 0])

        elif average == 'macro':
            # Compute average over precision of individual classes
            precision = 0
            for idx in range(self.num_classes):
                denom = self.confusion_matrices[idx, 0, 0] + self.confusion_matrices[idx, 1, 0]
                if denom != 0:
                    precision += (self.confusion_matrices[idx, 0, 0] / denom)
                else:
                    precision += 1
            precision /= self.num_classes

        return precision


    def print_all_measures(self) -> None:
        """ Evaluates and prints all the measures. """

        print('Accuracy:', self.accuracy_score())
        print('Precision (Micro):', self.precision_score(average='micro'))
        print('Recall (Micro):', self.recall_score(average='micro'))
        print('F1 Score (Micro):', self.f1_score(average='micro'))
        print('Precision (Macro):', self.precision_score(average='macro'))
        print('Recall (Macro):', self.recall_score(average='macro'))
        print('F1 Score (Macro):', self.f1_score(average='macro'))
